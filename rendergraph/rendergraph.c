#include "rendergraph.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef RENDERGRAPH_FEATURE_VULKAN

#define VK_NO_PROTOTYPES

#if defined(__linux__)
#define RG_PLATFORM_XLIB
#define VK_USE_PLATFORM_XLIB_KHR
#include <X11/Xlib.h>
#elif defined(_WIN32)
#define RG_PLATFORM_WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#else
#error Unsupported OS
#endif

#define VOLK_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "volk.h"

#define RG_MAX(a, b) ((a > b) ? (a) : (b))
#define RG_CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))
#define RG_LENGTH(array) (sizeof(array) / sizeof(array[0]))

enum {
    RG_FRAMES_IN_FLIGHT = 2,
    RG_MAX_SHADER_STAGES = 4,
    RG_MAX_ATTACHMENTS = 16,
    RG_MAX_COLOR_ATTACHMENTS = 16,
    RG_MAX_VERTEX_ATTRIBUTES = 8,
    RG_MAX_DESCRIPTOR_SETS = 8,
    RG_MAX_DESCRIPTOR_BINDINGS = 8,
    RG_MAX_DESCRIPTOR_TYPES = 8,
    RG_SETS_PER_PAGE = 32,
    RG_BUFFER_POOL_CHUNK_SIZE = 65536,

    RG_MAX_GRAPH_PASSES = 8,
    RG_MAX_GRAPH_RESOURCES = 8,
    RG_MAX_PASS_RESOURCES = 8,
};

#ifdef RENDERGRAPH_FEATURE_VALIDATION
static const char *RG_REQUIRED_INSTANCE_EXTENSIONS[1] = {
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
};
static const char *RG_REQUIRED_VALIDATION_LAYERS[1] = {
    "VK_LAYER_KHRONOS_validation",
};
#else
static const char *RG_EQUIRED_INSTANCE_EXTENSIONS[0] = {};
static const char *RG_EQUIRED_VALIDATION_LAYERS[0] = {};
#endif

static const char *RG_REQUIRED_DEVICE_EXTENSIONS[1] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#define VK_CHECK(result)                                                                 \
    do                                                                                   \
    {                                                                                    \
        if (result != VK_SUCCESS)                                                        \
        {                                                                                \
            fprintf(stderr, "%s:%u vulkan error: %d\n", __FILE__, __LINE__, result);     \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// Hashing {{{
static void fnvHashReset(uint64_t *hash)
{
    *hash = 14695981039346656037ULL;
}

static void fnvHashUpdate(uint64_t *hash, uint8_t *bytes, size_t count)
{
    for (uint64_t i = 0; i < count; ++i)
    {
        *hash = ((*hash) * 1099511628211) ^ bytes[i];
    }
}
// }}}

// Hashmap {{{
typedef struct RgHashmap
{
    uint64_t size;
    uint64_t *hashes;
    uint64_t *values;
} RgHashmap;

static void rgHashmapInit(RgHashmap *hashmap, size_t size);
static void rgHashmapDestroy(RgHashmap *hashmap);
static void rgHashmapGrow(RgHashmap *hashmap);
static void rgHashmapSet(RgHashmap *hashmap, uint64_t hash, uint64_t value);
static uint64_t *rgHashmapGet(RgHashmap *hashmap, uint64_t hash);

static void rgHashmapInit(RgHashmap *hashmap, size_t size)
{
    memset(hashmap, 0, sizeof(*hashmap));

    hashmap->size = size;
    assert(hashmap->size > 0);

    // Round up to nearest power of two
    hashmap->size -= 1;
    hashmap->size |= hashmap->size >> 1;
    hashmap->size |= hashmap->size >> 2;
    hashmap->size |= hashmap->size >> 4;
    hashmap->size |= hashmap->size >> 8;
    hashmap->size |= hashmap->size >> 16;
    hashmap->size |= hashmap->size >> 32;
    hashmap->size += 1;

    // Init memory
    hashmap->hashes = (uint64_t *)malloc(hashmap->size * sizeof(uint64_t));
    memset(hashmap->hashes, 0, hashmap->size * sizeof(uint64_t));
    hashmap->values = (uint64_t *)malloc(hashmap->size * sizeof(uint64_t));
    memset(hashmap->values, 0, hashmap->size * sizeof(uint64_t));
}

static void rgHashmapDestroy(RgHashmap *hashmap)
{
    free(hashmap->values);
    free(hashmap->hashes);
}

static void rgHashmapGrow(RgHashmap *hashmap)
{
    uint64_t old_size = hashmap->size;
    uint64_t *old_hashes = hashmap->hashes;
    uint64_t *old_values = hashmap->values;

    hashmap->size *= 2;
    hashmap->hashes = (uint64_t *)malloc(hashmap->size * sizeof(uint64_t));
    memset(hashmap->hashes, 0, hashmap->size * sizeof(uint64_t));
    hashmap->values = (uint64_t *)malloc(hashmap->size * sizeof(uint64_t));
    memset(hashmap->values, 0, hashmap->size * sizeof(uint64_t));

    for (uint64_t i = 0; i < old_size; i++)
    {
        if (old_hashes[i] != 0)
        {
            rgHashmapSet(hashmap, old_hashes[i], old_values[i]);
        }
    }

    free(old_hashes);
    free(old_values);
}

static void rgHashmapSet(RgHashmap *hashmap, uint64_t hash, uint64_t value)
{
    assert(hash != 0);

    uint64_t i = hash & (hashmap->size - 1); // hash % size
    uint64_t iters = 0;

    while ((hashmap->hashes[i] != hash) && hashmap->hashes[i] != 0 &&
           iters < hashmap->size)
    {
        i = (i + 1) & (hashmap->size - 1); // (i+1) % size
        iters += 1;
    }

    if (iters >= hashmap->size)
    {
        rgHashmapGrow(hashmap);
        rgHashmapSet(hashmap, hash, value);
        return;
    }

    hashmap->hashes[i] = hash;
    hashmap->values[i] = value;
}

static uint64_t *rgHashmapGet(RgHashmap *hashmap, uint64_t hash)
{
    uint64_t i = hash & (hashmap->size - 1); // hash % size
    uint64_t iters = 0;

    while ((hashmap->hashes[i] != hash) && hashmap->hashes[i] != 0 &&
           iters < hashmap->size)
    {
        i = (i + 1) & (hashmap->size - 1); // (i+1) % size
        iters += 1;
    }

    if (iters >= hashmap->size)
    {
        return NULL;
    }

    if (hashmap->hashes[i] != 0)
    {
        return &hashmap->values[i];
    }

    return NULL;
}
// }}}

// Types {{{
struct RgDevice
{
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_callback;

    VkPhysicalDeviceProperties physical_device_properties;
    VkPhysicalDeviceFeatures physical_device_features;
    VkQueueFamilyProperties *queue_family_properties;
    uint32_t num_queue_family_properties;

    VkPhysicalDevice physical_device;
    VkDevice device;
    VmaAllocator allocator;

    VkQueue graphics_queue;

    struct
    {
        uint32_t graphics;
        uint32_t compute;
    } queue_family_indices;

    VkCommandPool graphics_command_pool;
};

typedef struct RgBufferPool RgBufferPool;
typedef struct RgBufferChunk RgBufferChunk;

struct RgBufferChunk
{
    RgBufferPool *pool;
    RgBufferChunk *next;

    RgBuffer *buffer;
    size_t offset;
    size_t size;
    uint8_t *mapping;
};

struct RgBufferPool
{
    RgDevice *device;
    RgBufferChunk *base_chunk;
    size_t chunk_size;
    size_t alignment;
    RgBufferUsage usage;
};

typedef struct
{
    RgBuffer *buffer;
    uint8_t *mapping;
    size_t offset;
    size_t size;
} RgBufferAllocation;

typedef union
{
    VkDescriptorImageInfo image;
    VkDescriptorBufferInfo buffer;
} RgDescriptor;

struct RgCmdBuffer
{
    RgDevice *device;
    RgPass *current_pass;
    VkCommandBuffer cmd_buffer;

    RgPipeline *current_pipeline;
    RgDescriptor bound_descriptors[RG_MAX_DESCRIPTOR_SETS][RG_MAX_DESCRIPTOR_BINDINGS];
    uint64_t set_hashes[RG_MAX_DESCRIPTOR_SETS];

    RgBufferPool ubo_pool;
    RgBufferPool vbo_pool;
    RgBufferPool ibo_pool;
};

typedef struct RgSwapchain
{
    RgDevice *device;

    RgPlatformWindowInfo window;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;

    uint32_t present_family_index;
    VkQueue present_queue;

    uint32_t num_images;
    VkImage *images;
    VkImageView *image_views;

    VkFormat image_format;
    VkExtent2D extent;
    uint32_t current_image_index;
} RgSwapchain;

struct RgResource
{
    RgResourceType type;
    union
    {
        RgGraphImageInfo image_info;
        RgBufferInfo buffer_info;
    };
    union
    {
        RgImage *image;
        RgBuffer *buffer;
    };
};

struct RgPass
{
    RgGraph *graph;

    VkRenderPass renderpass;
    VkExtent2D extent;
    uint64_t hash;
    uint32_t num_attachments;
    uint32_t num_color_attachments;
    bool has_depth_attachment;
    uint32_t depth_attachment_index;

    VkFramebuffer *framebuffers;
    uint32_t num_framebuffers;

    uint32_t num_outputs;
    uint32_t outputs[RG_MAX_PASS_RESOURCES];

    uint32_t num_inputs;
    uint32_t inputs[RG_MAX_PASS_RESOURCES];

    VkFramebuffer current_framebuffer;

    RgPassCallback *callback;
};

struct RgGraph
{
    RgDevice *device;

    bool built;
    bool has_swapchain;
    RgSwapchain swapchain;

    void *user_data;

    RgNode *nodes;
    uint32_t num_nodes;

    RgPass passes[RG_MAX_GRAPH_PASSES];
    uint32_t num_passes;

    RgResource resources[RG_MAX_GRAPH_RESOURCES];
    uint32_t num_resources;

    VkSemaphore image_available_semaphores[RG_FRAMES_IN_FLIGHT];
    uint32_t current_frame;

    VkBufferMemoryBarrier *buffer_barriers;
    uint32_t num_buffer_barriers;

    VkImageMemoryBarrier *image_barriers;
    uint32_t num_image_barriers;
};

struct RgNode
{
    struct
    {
        RgCmdBuffer cmd_buffer;
        VkSemaphore execution_finished_semaphore;

        uint32_t num_wait_semaphores;
        VkSemaphore *wait_semaphores;
        VkPipelineStageFlags *wait_stages;

        VkFence fence;
    } frames[RG_FRAMES_IN_FLIGHT];

    uint32_t *pass_indices;
    uint32_t num_pass_indices;
};

typedef struct RgDescriptorPoolChunk
{
    struct RgDescriptorPoolChunk *next;
    VkDescriptorPool pool;
    VkDescriptorSet sets[RG_SETS_PER_PAGE];
    uint32_t allocated_count;
    RgHashmap map; // stores indices to allocated descriptor sets
} RgDescriptorPoolChunk;

typedef struct RgDescriptorPool
{
    RgDevice *device;
    RgDescriptorPoolChunk *base_chunk;
    VkDescriptorSetLayout set_layout;
    VkDescriptorUpdateTemplate update_template;
    VkDescriptorPoolSize pool_sizes[RG_MAX_DESCRIPTOR_TYPES];
    uint32_t num_pool_sizes;
    uint32_t num_bindings;
} RgDescriptorPool;

struct RgPipeline
{
    RgPipelineInfo info;
    RgHashmap instances;

    RgDescriptorPool pools[RG_MAX_DESCRIPTOR_SETS];
    uint32_t num_sets;

    VkPipelineLayout pipeline_layout;
    VkShaderModule vertex_shader;
    VkShaderModule fragment_shader;
};
// }}}

// Device memory allocator {{{
#if 0
struct rg_allocation_t
{
    VkDeviceMemory handle;
    uint32_t type;
    uint32_t id;
    size_t size;
    size_t offset;
};

static int32_t
rg_find_memory_properties(
        VkPhysicalDeviceMemoryProperties* memory_properties,
        uint32_t memory_type_bits_requirement,
        VkMemoryPropertyFlags required_properties)
{
    uint32_t memory_count = memory_properties->memoryTypeCount;

    for (uint32_t memory_index = 0; memory_index < memory_count; ++memory_index)
    {
        uint32_t memory_type_bits = (1 << memory_index);
        bool is_required_memory_type = memory_type_bits_requirement & memory_type_bits;

        VkMemoryPropertyFlags properties = memory_properties->memoryTypes[memory_index].propertyFlags;
        bool has_required_properties = (properties & required_properties) == required_properties;

        if (is_required_memory_type && has_required_properties)
        {
            return (int32_t)memory_index;
        }
    }

    // failed to find memory type
    return -1;
}
#endif
// }}}

// Device {{{
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_message_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
    fprintf(stderr, "Validation layer: %s\n", pCallbackData->pMessage);

    return VK_FALSE;
}

static bool check_validation_layer_support()
{
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, NULL);
    VkLayerProperties *available_layers =
        (VkLayerProperties *)malloc(sizeof(VkLayerProperties) * count);
    vkEnumerateInstanceLayerProperties(&count, available_layers);

    for (uint32_t i = 0; i < RG_LENGTH(RG_REQUIRED_VALIDATION_LAYERS); ++i)
    {
        const char *required_layer_name = RG_REQUIRED_VALIDATION_LAYERS[i];
        bool layer_found = false;

        for (uint32_t j = 0; j < count; ++j)
        {
            VkLayerProperties *layer = &available_layers[j];
            if (strcmp(layer->layerName, required_layer_name) == 0)
            {
                layer_found = true;
                break;
            }
        }

        if (!layer_found)
        {
            free(available_layers);
            return false;
        }
    }

    free(available_layers);
    return true;
}

static uint32_t get_queue_family_index(RgDevice *device, VkQueueFlagBits queue_flags)
{
    // Dedicated queue for compute
    // Try to find a queue family index that supports compute but not graphics
    if (queue_flags & VK_QUEUE_COMPUTE_BIT)
    {
        for (uint32_t i = 0; i < device->num_queue_family_properties; i++)
        {
            if ((device->queue_family_properties[i].queueFlags & queue_flags) &&
                ((device->queue_family_properties[i].queueFlags &
                  VK_QUEUE_GRAPHICS_BIT) == 0))
            {
                return i;
            }
        }
    }

    // For other queue types or if no separate compute queue is present,
    // return the first one to support the requested flags
    for (uint32_t i = 0; i < device->num_queue_family_properties; i++)
    {
        if (device->queue_family_properties[i].queueFlags & queue_flags)
        {
            return i;
        }
    }

    return UINT32_MAX;
}

RgDevice *rgDeviceCreate()
{
    RgDevice *device = (RgDevice *)malloc(sizeof(RgDevice));
    memset(device, 0, sizeof(*device));

    VK_CHECK(volkInitialize());

#ifdef RENDERGRAPH_FEATURE_VALIDATION
    if (check_validation_layer_support())
    {
        fprintf(stderr, "Using validation layers\n");
    }
    else
    {
        fprintf(stderr, "Validation layers requested but not available\n");
    }
#endif

    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = NULL;
    app_info.pApplicationName = "Rendergraph application";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "Rendergraph";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.flags = 0;
    instance_info.pApplicationInfo = &app_info;

    instance_info.enabledLayerCount = RG_LENGTH(RG_REQUIRED_VALIDATION_LAYERS);
    instance_info.ppEnabledLayerNames = RG_REQUIRED_VALIDATION_LAYERS;

#if defined(RG_PLATFORM_XLIB)
    const char *platform_extensions[2] = {
        "VK_KHR_surface",
        "VK_KHR_xlib_surface",
    };
#elif defined(RG_PLATFORM_WIN32)
    const char *platform_extensions[2] = {
        "VK_KHR_surface",
        "VK_KHR_win32_surface",
    };
#else
    const char *platform_extensions[0];
#endif

    uint32_t num_instance_extensions = 0;
    num_instance_extensions += RG_LENGTH(platform_extensions);
    num_instance_extensions += RG_LENGTH(RG_REQUIRED_INSTANCE_EXTENSIONS);

    char **instance_extensions =
        (char **)malloc(num_instance_extensions * sizeof(*instance_extensions));

    memcpy(
        instance_extensions,
        platform_extensions,
        RG_LENGTH(platform_extensions) * sizeof(char *));
    memcpy(
        instance_extensions + RG_LENGTH(platform_extensions),
        RG_REQUIRED_INSTANCE_EXTENSIONS,
        RG_LENGTH(RG_REQUIRED_INSTANCE_EXTENSIONS) * sizeof(char *));

    instance_info.enabledExtensionCount = num_instance_extensions;
    instance_info.ppEnabledExtensionNames = (const char *const *)instance_extensions;

    VK_CHECK(vkCreateInstance(&instance_info, NULL, &device->instance));

    volkLoadInstance(device->instance);

    free(instance_extensions);

#ifdef RENDERGRAPH_FEATURE_VALIDATION
    VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
    memset(&debug_create_info, 0, sizeof(debug_create_info));
    debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_create_info.pfnUserCallback = &debug_message_callback;

    VK_CHECK(vkCreateDebugUtilsMessengerEXT(
        device->instance, &debug_create_info, NULL, &device->debug_callback));
#endif

    uint32_t num_physical_devices = 0;
    vkEnumeratePhysicalDevices(device->instance, &num_physical_devices, NULL);
    VkPhysicalDevice *physical_devices =
        (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * num_physical_devices);
    vkEnumeratePhysicalDevices(device->instance, &num_physical_devices, physical_devices);

    if (num_physical_devices == 0)
    {
        fprintf(stderr, "No physical devices found\n");
        exit(1);
    }

    device->physical_device = physical_devices[0];

    free(physical_devices);

    vkGetPhysicalDeviceProperties(
        device->physical_device, &device->physical_device_properties);
    vkGetPhysicalDeviceFeatures(
        device->physical_device, &device->physical_device_features);

    vkGetPhysicalDeviceQueueFamilyProperties(
        device->physical_device, &device->num_queue_family_properties, NULL);
    device->queue_family_properties = (VkQueueFamilyProperties *)malloc(
        sizeof(VkQueueFamilyProperties) * device->num_queue_family_properties);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device->physical_device,
        &device->num_queue_family_properties,
        device->queue_family_properties);

    fprintf(
        stderr,
        "Using physical device: %s\n",
        device->physical_device_properties.deviceName);
    VkPhysicalDeviceFeatures enabled_features;
    memset(&enabled_features, 0, sizeof(enabled_features));

    if (device->physical_device_features.samplerAnisotropy)
    {
        enabled_features.samplerAnisotropy = VK_TRUE;
    }

    if (device->physical_device_features.fillModeNonSolid)
    {
        enabled_features.fillModeNonSolid = VK_TRUE;
    }

    VkQueueFlags requested_queue_types = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;

    VkDeviceQueueCreateInfo queue_create_infos[2];
    uint32_t num_queue_create_infos = 0;

    // Get queue family indices for the requested queue family types
    // Note that the indices may overlap depending on the implementation

    const float default_queue_priority = 0.0f;

    // Graphics queue
    if (requested_queue_types & VK_QUEUE_GRAPHICS_BIT)
    {
        device->queue_family_indices.graphics =
            get_queue_family_index(device, VK_QUEUE_GRAPHICS_BIT);

        VkDeviceQueueCreateInfo queue_info;
        memset(&queue_info, 0, sizeof(queue_info));
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = device->queue_family_indices.graphics;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &default_queue_priority;

        queue_create_infos[num_queue_create_infos++] = queue_info;
    }
    else
    {
        device->queue_family_indices.graphics = 0;
    }

    // Dedicated compute queue
    if (requested_queue_types & VK_QUEUE_COMPUTE_BIT)
    {
        device->queue_family_indices.compute =
            get_queue_family_index(device, VK_QUEUE_COMPUTE_BIT);
        if (device->queue_family_indices.compute != device->queue_family_indices.graphics)
        {
            // If compute family index differs,
            // we need an additional queue create info for the compute queue
            VkDeviceQueueCreateInfo queue_info;
            memset(&queue_info, 0, sizeof(queue_info));
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = device->queue_family_indices.compute;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &default_queue_priority;

            queue_create_infos[num_queue_create_infos++] = queue_info;
        }
    }
    else
    {
        // Else we use the same queue
        device->queue_family_indices.compute = device->queue_family_indices.graphics;
    }

    // Create the logical device representation
    VkDeviceCreateInfo device_create_info;
    memset(&device_create_info, 0, sizeof(device_create_info));
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = num_queue_create_infos;
    device_create_info.pQueueCreateInfos = queue_create_infos;
    device_create_info.pEnabledFeatures = &enabled_features;

    uint32_t num_device_extensions = 0;
    vkEnumerateDeviceExtensionProperties(
        device->physical_device, NULL, &num_device_extensions, NULL);
    VkExtensionProperties *device_extensions = (VkExtensionProperties *)malloc(
        sizeof(VkExtensionProperties) * num_device_extensions);
    vkEnumerateDeviceExtensionProperties(
        device->physical_device, NULL, &num_device_extensions, device_extensions);

    uint32_t num_enabled_device_extensions = RG_LENGTH(RG_REQUIRED_DEVICE_EXTENSIONS);
    char **enabled_device_extensions =
        (char **)malloc(sizeof(char *) * num_enabled_device_extensions);

    memcpy(
        enabled_device_extensions,
        RG_REQUIRED_DEVICE_EXTENSIONS,
        RG_LENGTH(RG_REQUIRED_DEVICE_EXTENSIONS) * sizeof(char *));

    device_create_info.enabledExtensionCount = num_enabled_device_extensions;
    device_create_info.ppEnabledExtensionNames =
        (const char *const *)enabled_device_extensions;

    VK_CHECK(vkCreateDevice(
        device->physical_device, &device_create_info, NULL, &device->device));

    free(enabled_device_extensions);
    free(device_extensions);

    //
    // Initialize VMA
    //
    VmaVulkanFunctions vk_funcs = {0};
    vk_funcs.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vk_funcs.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vk_funcs.vkAllocateMemory = vkAllocateMemory;
    vk_funcs.vkFreeMemory = vkFreeMemory;
    vk_funcs.vkMapMemory = vkMapMemory;
    vk_funcs.vkUnmapMemory = vkUnmapMemory;
    vk_funcs.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vk_funcs.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    vk_funcs.vkBindBufferMemory = vkBindBufferMemory;
    vk_funcs.vkBindImageMemory = vkBindImageMemory;
    vk_funcs.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    vk_funcs.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vk_funcs.vkCreateBuffer = vkCreateBuffer;
    vk_funcs.vkDestroyBuffer = vkDestroyBuffer;
    vk_funcs.vkCreateImage = vkCreateImage;
    vk_funcs.vkDestroyImage = vkDestroyImage;
    vk_funcs.vkCmdCopyBuffer = vkCmdCopyBuffer;

    VmaAllocatorCreateInfo allocator_info = {0};
    allocator_info.physicalDevice = device->physical_device;
    allocator_info.device = device->device;
    allocator_info.instance = device->instance;
    allocator_info.pVulkanFunctions = &vk_funcs;

    VK_CHECK(vmaCreateAllocator(&allocator_info, &device->allocator));

    vkGetDeviceQueue(
        device->device,
        device->queue_family_indices.graphics,
        0,
        &device->graphics_queue);

    VkCommandPoolCreateInfo cmd_pool_info;
    memset(&cmd_pool_info, 0, sizeof(cmd_pool_info));
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.queueFamilyIndex = device->queue_family_indices.graphics;
    cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(
        device->device, &cmd_pool_info, NULL, &device->graphics_command_pool));

    return device;
}

void rgDeviceDestroy(RgDevice *device)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));

    vkDestroyCommandPool(device->device, device->graphics_command_pool, NULL);
    vmaDestroyAllocator(device->allocator);
    vkDestroyDevice(device->device, NULL);

#ifdef RENDERGRAPH_FEATURE_VALIDATION
    vkDestroyDebugUtilsMessengerEXT(device->instance, device->debug_callback, NULL);
#endif

    vkDestroyInstance(device->instance, NULL);

    free(device->queue_family_properties);

    free(device);
}
// }}}

// Swapchain setup {{{
static void
rgGetWindowSize(RgPlatformWindowInfo *window, uint32_t *width, uint32_t *height)
{
#if defined(RG_PLATFORM_XLIB)
    XWindowAttributes attribs;
    XGetWindowAttributes(
        (Display *)window->x11.display, (Window)window->x11.window, &attribs);

    if (width) *width = (uint32_t)attribs.width;
    if (height) *height = (uint32_t)attribs.height;
#elif defined(RG_PLATFORM_WIN32)
    RECT area;
    GetClientRect(window->win32.window, &area);

    if (width) *width = (uint32_t)area.right;
    if (height) *height = (uint32_t)area.bottom;
#endif
}

static void
rgSwapchainInit(RgDevice *device, RgSwapchain *swapchain, RgPlatformWindowInfo *window)
{
    memset(swapchain, 0, sizeof(*swapchain));

    if (window)
    {
        swapchain->window = *window;
    }
    swapchain->device = device;

#if defined(RG_PLATFORM_XLIB)
    VkXlibSurfaceCreateInfoKHR surface_ci;
    memset(&surface_ci, 0, sizeof(surface_ci));
    surface_ci.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
    surface_ci.dpy = (Display *)window->x11.display;
    surface_ci.window = (Window)window->x11.window;

    VK_CHECK(
        vkCreateXlibSurfaceKHR(device->instance, &surface_ci, NULL, &swapchain->surface));
#elif defined(RG_PLATFORM_WIN32)
    VkWin32SurfaceCreateInfoKHR surface_ci;
    memset(&surface_ci, 0, sizeof(surface_ci));
    surface_ci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surface_ci.hinstance = GetModuleHandle(NULL);
    surface_ci.hwnd = (HWND)window->win32.window;

    VK_CHECK(vkCreateWin32SurfaceKHR(
        device->instance, &surface_ci, NULL, &swapchain->surface));
#endif

    swapchain->present_family_index = UINT32_MAX;

    uint32_t num_queue_families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        device->physical_device, &num_queue_families, NULL);
    VkQueueFamilyProperties *queue_families = (VkQueueFamilyProperties *)malloc(
        sizeof(VkQueueFamilyProperties) * num_queue_families);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device->physical_device, &num_queue_families, queue_families);

    for (uint32_t i = 0; i < num_queue_families; ++i)
    {
        VkQueueFamilyProperties *queue_family = &queue_families[i];

        VkBool32 supported;
        vkGetPhysicalDeviceSurfaceSupportKHR(
            device->physical_device, i, swapchain->surface, &supported);

        if (queue_family->queueCount > 0 && supported)
        {
            swapchain->present_family_index = i;
            break;
        }
    }

    free(queue_families);

    if (swapchain->present_family_index == UINT32_MAX)
    {
        fprintf(stderr, "Could not obtain a present queue family.\n");
        exit(1);
    }

    // Get present queue
    vkGetDeviceQueue(
        device->device, swapchain->present_family_index, 0, &swapchain->present_queue);
}

static void rgSwapchainDestroy(RgDevice *device, RgSwapchain *swapchain)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));

    for (uint32_t i = 0; i < swapchain->num_images; ++i)
    {
        vkDestroyImageView(device->device, swapchain->image_views[i], NULL);
        swapchain->image_views[i] = VK_NULL_HANDLE;
    }

    vkDestroySwapchainKHR(device->device, swapchain->swapchain, NULL);
    vkDestroySurfaceKHR(device->instance, swapchain->surface, NULL);

    free(swapchain->images);
    free(swapchain->image_views);
}

static void rgSwapchainResize(RgSwapchain *swapchain)
{
    uint32_t width, height;
    rgGetWindowSize(&swapchain->window, &width, &height);

    VK_CHECK(vkDeviceWaitIdle(swapchain->device->device));

    // Destroy old stuff first
    {
        for (uint32_t i = 0; i < swapchain->num_images; ++i)
        {
            vkDestroyImageView(
                swapchain->device->device, swapchain->image_views[i], NULL);
            swapchain->image_views[i] = VK_NULL_HANDLE;
        }

        if (swapchain->swapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(swapchain->device->device, swapchain->swapchain, NULL);
            swapchain->swapchain = VK_NULL_HANDLE;
        }
    }

    // Get capabilities
    VkSurfaceCapabilitiesKHR capabilities = {0};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        swapchain->device->physical_device, swapchain->surface, &capabilities);

    // Get format
    VkSurfaceFormatKHR surface_format;
    memset(&surface_format, 0, sizeof(surface_format));
    {
        uint32_t num_formats = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            swapchain->device->physical_device, swapchain->surface, &num_formats, NULL);
        VkSurfaceFormatKHR *formats =
            (VkSurfaceFormatKHR *)malloc(sizeof(VkSurfaceFormatKHR) * num_formats);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            swapchain->device->physical_device,
            swapchain->surface,
            &num_formats,
            formats);

        if (num_formats == 0)
        {
            printf("Physical device does not support swapchain creation\n");
            exit(1);
        }

        if (num_formats == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
        {
            surface_format.format = VK_FORMAT_B8G8R8A8_UNORM;
            surface_format.colorSpace = formats[0].colorSpace;
        }

        for (uint32_t i = 0; i < num_formats; ++i)
        {
            VkSurfaceFormatKHR *format = &formats[i];
            if (format->format == VK_FORMAT_B8G8R8A8_UNORM)
            {
                surface_format = *format;
                break;
            }
        }

        free(formats);
    }

    swapchain->image_format = surface_format.format;

    // Get present mode
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    {
        uint32_t num_present_modes = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            swapchain->device->physical_device,
            swapchain->surface,
            &num_present_modes,
            NULL);
        VkPresentModeKHR *present_modes =
            (VkPresentModeKHR *)malloc(sizeof(VkPresentModeKHR) * num_present_modes);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            swapchain->device->physical_device,
            swapchain->surface,
            &num_present_modes,
            present_modes);

        if (num_present_modes == 0)
        {
            fprintf(stderr, "Physical device does not support swapchain creation\n");
            exit(1);
        }

        for (uint32_t i = 0; i < num_present_modes; ++i)
        {
            if (present_modes[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
                present_mode = present_modes[i];
        }

        for (uint32_t i = 0; i < num_present_modes; ++i)
        {
            if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
                present_mode = present_modes[i];
        }

        for (uint32_t i = 0; i < num_present_modes; ++i)
        {
            if (present_modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
                present_mode = present_modes[i];
        }

        free(present_modes);
    }

    width = RG_CLAMP(
        width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    height = RG_CLAMP(
        height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    swapchain->extent = (VkExtent2D){width, height};

    swapchain->num_images = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 &&
        swapchain->num_images > capabilities.maxImageCount)
    {
        swapchain->num_images = capabilities.maxImageCount;
    }

    VkImageUsageFlags image_usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    {
        fprintf(
            stderr,
            "Physical device does not support "
            "VK_IMAGE_USAGE_TRANSFER_DST_BIT in swapchains\n");
        exit(1);
    }

    VkSwapchainCreateInfoKHR create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = swapchain->surface;
    create_info.minImageCount = swapchain->num_images;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = swapchain->extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = image_usage;

    uint32_t queue_family_indices[2] = {
        swapchain->device->queue_family_indices.graphics,
        swapchain->present_family_index};

    if (swapchain->device->queue_family_indices.graphics !=
        swapchain->present_family_index)
    {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = queue_family_indices;
    }
    else
    {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = 0;
        create_info.pQueueFamilyIndices = NULL;
    }

    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;

    VkSwapchainKHR old_swapchain = swapchain->swapchain;
    create_info.oldSwapchain = old_swapchain;

    VK_CHECK(vkCreateSwapchainKHR(
        swapchain->device->device, &create_info, NULL, &swapchain->swapchain));

    vkGetSwapchainImagesKHR(
        swapchain->device->device, swapchain->swapchain, &swapchain->num_images, NULL);
    swapchain->images =
        (VkImage *)realloc(swapchain->images, sizeof(VkImage) * swapchain->num_images);
    vkGetSwapchainImagesKHR(
        swapchain->device->device,
        swapchain->swapchain,
        &swapchain->num_images,
        swapchain->images);

    swapchain->image_views = (VkImageView *)realloc(
        swapchain->image_views, sizeof(VkImageView) * swapchain->num_images);
    for (size_t i = 0; i < swapchain->num_images; i++)
    {
        VkImageViewCreateInfo view_create_info;
        memset(&view_create_info, 0, sizeof(view_create_info));
        view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_create_info.image = swapchain->images[i];
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_create_info.format = swapchain->image_format;
        view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_create_info.subresourceRange.baseMipLevel = 0;
        view_create_info.subresourceRange.levelCount = 1;
        view_create_info.subresourceRange.baseArrayLayer = 0;
        view_create_info.subresourceRange.layerCount = 1;

        VK_CHECK(vkCreateImageView(
            swapchain->device->device,
            &view_create_info,
            NULL,
            &swapchain->image_views[i]));
    }
}
// }}}

// Type conversions {{{
static VkFormat format_to_vk(RgFormat fmt)
{
    switch (fmt)
    {
    case RG_FORMAT_UNDEFINED: return VK_FORMAT_UNDEFINED;

    case RG_FORMAT_RGB8_UNORM: return VK_FORMAT_R8G8B8_UNORM;
    case RG_FORMAT_RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;

    case RG_FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;

    case RG_FORMAT_R32_SFLOAT: return VK_FORMAT_R32_SFLOAT;
    case RG_FORMAT_RG32_SFLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case RG_FORMAT_RGB32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case RG_FORMAT_RGBA32_SFLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;

    case RG_FORMAT_RGBA16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;

    case RG_FORMAT_D32_SFLOAT: return VK_FORMAT_D32_SFLOAT;
    case RG_FORMAT_D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
    }
    assert(0);
    return 0;
}

static VkFilter filter_to_vk(RgFilter value)
{
    switch (value)
    {
    case RG_FILTER_LINEAR: return VK_FILTER_LINEAR;
    case RG_FILTER_NEAREST: return VK_FILTER_NEAREST;
    }
    assert(0);
    return 0;
}

static VkSamplerAddressMode address_mode_to_vk(RgSamplerAddressMode value)
{
    switch (value)
    {
    case RG_SAMPLER_ADDRESS_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case RG_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case RG_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE:
        return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    }
    assert(0);
    return 0;
}

static VkBorderColor border_color_to_vk(RgBorderColor value)
{
    switch (value)
    {
    case RG_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
        return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    case RG_BORDER_COLOR_INT_TRANSPARENT_BLACK:
        return VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
    case RG_BORDER_COLOR_FLOAT_OPAQUE_BLACK: return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    case RG_BORDER_COLOR_INT_OPAQUE_BLACK: return VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    case RG_BORDER_COLOR_FLOAT_OPAQUE_WHITE: return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    case RG_BORDER_COLOR_INT_OPAQUE_WHITE: return VK_BORDER_COLOR_INT_OPAQUE_WHITE;
    }
    assert(0);
    return 0;
}

static VkIndexType index_type_to_vk(RgIndexType index_type)
{
    switch (index_type)
    {
    case RG_INDEX_TYPE_UINT16: return VK_INDEX_TYPE_UINT16;
    case RG_INDEX_TYPE_UINT32: return VK_INDEX_TYPE_UINT32;
    }
    assert(0);
    return 0;
}

static VkCullModeFlagBits cull_mode_to_vk(RgCullMode cull_mode)
{
    switch (cull_mode)
    {
    case RG_CULL_MODE_NONE: return VK_CULL_MODE_NONE;
    case RG_CULL_MODE_BACK: return VK_CULL_MODE_BACK_BIT;
    case RG_CULL_MODE_FRONT: return VK_CULL_MODE_FRONT_BIT;
    case RG_CULL_MODE_FRONT_AND_BACK: return VK_CULL_MODE_FRONT_AND_BACK;
    }
    assert(0);
    return 0;
}

static VkFrontFace front_face_to_vk(RgFrontFace front_face)
{
    switch (front_face)
    {
    case RG_FRONT_FACE_CLOCKWISE: return VK_FRONT_FACE_CLOCKWISE;
    case RG_FRONT_FACE_COUNTER_CLOCKWISE: return VK_FRONT_FACE_COUNTER_CLOCKWISE;
    }
    assert(0);
    return 0;
}

static VkPolygonMode polygon_mode_to_vk(RgPolygonMode polygon_mode)
{
    switch (polygon_mode)
    {
    case RG_POLYGON_MODE_FILL: return VK_POLYGON_MODE_FILL;
    case RG_POLYGON_MODE_LINE: return VK_POLYGON_MODE_LINE;
    case RG_POLYGON_MODE_POINT: return VK_POLYGON_MODE_POINT;
    }
    assert(0);
    return 0;
}

static VkPrimitiveTopology primitive_topology_to_vk(RgPrimitiveTopology value)
{
    switch (value)
    {
    case RG_PRIMITIVE_TOPOLOGY_LINE_LIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case RG_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
    assert(0);
    return 0;
}

static VkDescriptorType pipeline_binding_type_to_vk(RgPipelineBindingType type)
{
    switch (type)
    {
    case RG_BINDING_UNIFORM_BUFFER: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case RG_BINDING_STORAGE_BUFFER: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case RG_BINDING_IMAGE: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case RG_BINDING_SAMPLER: return VK_DESCRIPTOR_TYPE_SAMPLER;
    case RG_BINDING_IMAGE_SAMPLER: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    assert(0);
    return 0;
}
// }}}

// Buffer {{{
struct RgBuffer
{
    RgBufferInfo info;
    VkBuffer buffer;
    VmaAllocation allocation;
};

RgBuffer *rgBufferCreate(RgDevice *device, RgBufferInfo *info)
{
    RgBuffer *buffer = (RgBuffer *)malloc(sizeof(RgBuffer));
    memset(buffer, 0, sizeof(*buffer));

    buffer->info = *info;

    assert(buffer->info.size > 0);
    assert(buffer->info.memory > 0);
    assert(buffer->info.usage > 0);

    VkBufferCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = buffer->info.size;

    if (buffer->info.usage & RG_BUFFER_USAGE_VERTEX)
        ci.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_INDEX)
        ci.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_UNIFORM)
        ci.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_TRANSFER_SRC)
        ci.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_TRANSFER_DST)
        ci.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_STORAGE)
        ci.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo alloc_info = {0};
    alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    switch (buffer->info.memory)
    {
    case RG_BUFFER_MEMORY_HOST:
        alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    case RG_BUFFER_MEMORY_DEVICE:
        alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    }

    VK_CHECK(vmaCreateBuffer(
        device->allocator, &ci, &alloc_info, &buffer->buffer, &buffer->allocation, NULL));

    return buffer;
}

void rgBufferDestroy(RgDevice *device, RgBuffer *buffer)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));
    if (buffer->buffer)
    {
        vmaDestroyBuffer(device->allocator, buffer->buffer, buffer->allocation);
    }
    buffer->buffer = VK_NULL_HANDLE;
    buffer->allocation = VK_NULL_HANDLE;

    free(buffer);
}

void *rgBufferMap(RgDevice *device, RgBuffer *buffer)
{
    void *ptr;
    VK_CHECK(vmaMapMemory(device->allocator, buffer->allocation, &ptr));
    return ptr;
}

void rgBufferUnmap(RgDevice *device, RgBuffer *buffer)
{
    vmaUnmapMemory(device->allocator, buffer->allocation);
}

void rgBufferUpload(
    RgDevice *device, RgBuffer *buffer, size_t offset, size_t size, void *data)
{
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    RgBufferInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.size = size;
    buffer_info.usage = RG_BUFFER_USAGE_TRANSFER_SRC;
    buffer_info.memory = RG_BUFFER_MEMORY_HOST;

    RgBuffer *staging = rgBufferCreate(device, &buffer_info);

    void *staging_ptr = rgBufferMap(device, staging);
    memcpy(staging_ptr, data, size);
    rgBufferUnmap(device, staging);

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device->device, &fence_info, NULL, &fence));

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = device->graphics_command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device->device, &alloc_info, &cmd_buffer));

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));

    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.srcOffset = 0;
    region.dstOffset = offset;
    region.size = size;
    vkCmdCopyBuffer(cmd_buffer, staging->buffer, buffer->buffer, 1, &region);

    VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_buffer;

    VK_CHECK(vkQueueSubmit(device->graphics_queue, 1, &submit, fence));

    VK_CHECK(vkWaitForFences(device->device, 1, &fence, VK_TRUE, 1 * 1000000000ULL));
    vkDestroyFence(device->device, fence, NULL);

    vkFreeCommandBuffers(device->device, device->graphics_command_pool, 1, &cmd_buffer);

    rgBufferDestroy(device, staging);
}
// }}}

// Image {{{
struct RgImage
{
    RgImageInfo info;
    VkImage image;
    VmaAllocation allocation;
    VkImageView view;
    VkImageAspectFlags aspect;
};

RgImage *rgImageCreate(RgDevice *device, RgImageInfo *info)
{
    RgImage *image = (RgImage *)malloc(sizeof(RgImage));
    memset(image, 0, sizeof(*image));

    image->info = *info;

    if (image->info.depth == 0) image->info.depth = 1;
    if (image->info.sample_count == 0) image->info.sample_count = 1;
    if (image->info.mip_count == 0) image->info.mip_count = 1;
    if (image->info.layer_count == 0) image->info.layer_count = 1;
    if (image->info.usage == 0)
        image->info.usage = RG_IMAGE_USAGE_SAMPLED | RG_IMAGE_USAGE_TRANSFER_DST;
    if (image->info.aspect == 0) image->info.aspect = RG_IMAGE_ASPECT_COLOR;

    assert(image->info.width > 0);
    assert(image->info.height > 0);
    assert(image->info.format != RG_FORMAT_UNDEFINED);

    {
        VkImageCreateInfo ci;
        memset(&ci, 0, sizeof(ci));
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format_to_vk(image->info.format);
        ci.extent.width = image->info.width;
        ci.extent.height = image->info.height;
        ci.extent.depth = image->info.depth;
        ci.mipLevels = image->info.mip_count;
        ci.arrayLayers = image->info.layer_count;
        ci.samples = (VkSampleCountFlagBits)image->info.sample_count;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;

        if (image->info.layer_count == 6)
        {
            ci.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        }

        if (image->info.usage & RG_IMAGE_USAGE_SAMPLED)
            ci.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_TRANSFER_DST)
            ci.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_TRANSFER_SRC)
            ci.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_STORAGE)
            ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_COLOR_ATTACHMENT)
            ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT)
            ci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        VmaAllocationCreateInfo alloc_create_info = {0};
        alloc_create_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VK_CHECK(vmaCreateImage(
            device->allocator,
            &ci,
            &alloc_create_info,
            &image->image,
            &image->allocation,
            NULL));
    }

    {
        VkImageViewCreateInfo ci;
        memset(&ci, 0, sizeof(ci));
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = image->image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = format_to_vk(image->info.format);
        ci.subresourceRange.baseMipLevel = 0;
        ci.subresourceRange.levelCount = image->info.mip_count;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount = image->info.layer_count;

        if (image->info.layer_count == 6)
        {
            ci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        }

        if (image->info.aspect & RG_IMAGE_ASPECT_COLOR)
            image->aspect |= VK_IMAGE_ASPECT_COLOR_BIT;
        if (image->info.aspect & RG_IMAGE_ASPECT_DEPTH)
            image->aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (image->info.aspect & RG_IMAGE_ASPECT_STENCIL)
            image->aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;

        ci.subresourceRange.aspectMask = image->aspect;

        VK_CHECK(vkCreateImageView(device->device, &ci, NULL, &image->view));
    }

    return image;
}

void rgImageDestroy(RgDevice *device, RgImage *image)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));
    if (image->view)
    {
        vkDestroyImageView(device->device, image->view, NULL);
    }
    if (image->image)
    {
        vmaDestroyImage(device->allocator, image->image, image->allocation);
    }
    image->view = VK_NULL_HANDLE;
    image->image = VK_NULL_HANDLE;
    image->allocation = VK_NULL_HANDLE;

    free(image);
}

void rgImageUpload(
    RgDevice *device, RgImageCopy *dst, RgExtent3D *extent, size_t size, void *data)
{
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    RgBufferInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.size = size;
    buffer_info.usage = RG_BUFFER_USAGE_TRANSFER_SRC;
    buffer_info.memory = RG_BUFFER_MEMORY_HOST;

    RgBuffer *staging = rgBufferCreate(device, &buffer_info);

    void *staging_ptr = rgBufferMap(device, staging);
    memcpy(staging_ptr, data, size);
    rgBufferUnmap(device, staging);

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device->device, &fence_info, NULL, &fence));

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = device->graphics_command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device->device, &alloc_info, &cmd_buffer));

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));

    VkImageSubresourceRange subresource_range;
    memset(&subresource_range, 0, sizeof(subresource_range));
    subresource_range.aspectMask = dst->image->aspect;
    subresource_range.baseMipLevel = dst->mip_level;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = dst->array_layer;
    subresource_range.layerCount = 1;

    VkImageMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.image = dst->image->image;
    barrier.subresourceRange = subresource_range;

    vkCmdPipelineBarrier(
        cmd_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &barrier);

    VkBufferImageCopy region;
    memset(&region, 0, sizeof(region));
    region.imageSubresource.aspectMask = dst->image->aspect;
    region.imageSubresource.mipLevel = dst->mip_level;
    region.imageSubresource.baseArrayLayer = dst->array_layer;
    region.imageSubresource.layerCount = 1;
    region.imageOffset.x = dst->offset.x;
    region.imageOffset.y = dst->offset.y;
    region.imageOffset.z = dst->offset.z;
    region.imageExtent.width = extent->width;
    region.imageExtent.height = extent->height;
    region.imageExtent.depth = extent->depth;

    vkCmdCopyBufferToImage(
        cmd_buffer,
        staging->buffer,
        dst->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);

    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    vkCmdPipelineBarrier(
        cmd_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &barrier);

    VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_buffer;

    VK_CHECK(vkQueueSubmit(device->graphics_queue, 1, &submit, fence));

    VK_CHECK(vkWaitForFences(device->device, 1, &fence, VK_TRUE, 1 * 1000000000ULL));
    vkDestroyFence(device->device, fence, NULL);

    vkFreeCommandBuffers(device->device, device->graphics_command_pool, 1, &cmd_buffer);

    rgBufferDestroy(device, staging);
}
// }}}

// Sampler {{{
struct RgSampler
{
    RgSamplerInfo info;
    VkSampler sampler;
};

RgSampler *rgSamplerCreate(RgDevice *device, RgSamplerInfo *info)
{
    RgSampler *sampler = (RgSampler *)malloc(sizeof(RgSampler));
    memset(sampler, 0, sizeof(*sampler));

    sampler->info = *info;

    if (sampler->info.min_lod == 0.0f && sampler->info.max_lod == 0.0f)
    {
        sampler->info.max_lod = 1.0f;
    }

    assert(sampler->info.max_lod >= sampler->info.min_lod);

    VkSamplerCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter = filter_to_vk(sampler->info.mag_filter);
    ci.minFilter = filter_to_vk(sampler->info.min_filter);
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU = address_mode_to_vk(sampler->info.address_mode);
    ci.addressModeV = address_mode_to_vk(sampler->info.address_mode);
    ci.addressModeW = address_mode_to_vk(sampler->info.address_mode);
    ci.minLod = sampler->info.min_lod;
    ci.maxLod = sampler->info.max_lod;
    ci.maxAnisotropy = 1.0f;
    ci.anisotropyEnable = (VkBool32)sampler->info.anisotropy;
    ci.borderColor = border_color_to_vk(sampler->info.border_color);
    VK_CHECK(vkCreateSampler(device->device, &ci, NULL, &sampler->sampler));

    return sampler;
}

void rgSamplerDestroy(RgDevice *device, RgSampler *sampler)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));
    if (sampler->sampler)
    {
        vkDestroySampler(device->device, sampler->sampler, NULL);
    }
    sampler->sampler = VK_NULL_HANDLE;

    free(sampler);
}
// }}}

// Buffer allocator {{{
static RgBufferChunk *rgBufferChunkCreate(RgBufferPool *pool, size_t minimum_size)
{
    RgBufferChunk *chunk = (RgBufferChunk *)malloc(sizeof(*chunk));
    memset(chunk, 0, sizeof(*chunk));

    chunk->pool = pool;
    chunk->size = RG_MAX(minimum_size, pool->chunk_size);

    RgBufferInfo buffer_info = {0};
    buffer_info.size = chunk->size;
    buffer_info.usage = pool->usage;
    buffer_info.memory = RG_BUFFER_MEMORY_HOST;
    chunk->buffer = rgBufferCreate(chunk->pool->device, &buffer_info);

    chunk->mapping = (uint8_t *)rgBufferMap(pool->device, chunk->buffer);

    return chunk;
}

static void rgBufferChunkDestroy(RgBufferChunk *chunk)
{
    if (!chunk) return;
    rgBufferChunkDestroy(chunk->next);

    rgBufferUnmap(chunk->pool->device, chunk->buffer);
    rgBufferDestroy(chunk->pool->device, chunk->buffer);
    free(chunk);
}

static void rgBufferPoolInit(
    RgDevice *device,
    RgBufferPool *pool,
    size_t chunk_size,
    size_t alignment,
    RgBufferUsage usage)
{
    memset(pool, 0, sizeof(*pool));
    pool->device = device;
    pool->chunk_size = chunk_size;
    pool->alignment = alignment;
    pool->usage = usage;
}

static void rgBufferPoolReset(RgBufferPool *pool)
{
    RgBufferChunk *chunk = pool->base_chunk;
    while (chunk)
    {
        chunk->offset = 0;
        chunk = chunk->next;
    }
}

static RgBufferAllocation rgBufferPoolAllocate(RgBufferPool *pool, size_t allocate_size)
{
    RgBufferAllocation alloc = {0};

    RgBufferChunk *chunk = pool->base_chunk;
    RgBufferChunk *last_chunk = NULL;

buffer_pool_allocate_use_block:
    while (chunk)
    {
        size_t aligned_offset =
            (chunk->offset + pool->alignment - 1) & ~(pool->alignment - 1);
        if (chunk->mapping != NULL && chunk->size >= aligned_offset + allocate_size)
        {
            // Found chunk that fits the allocation
            assert(aligned_offset % pool->alignment == 0);
            chunk->offset = aligned_offset + allocate_size;

            alloc.buffer = chunk->buffer;
            alloc.mapping = chunk->mapping + aligned_offset;
            alloc.offset = aligned_offset;
            alloc.size = allocate_size;
            return alloc;
        }

        last_chunk = chunk;
        chunk = chunk->next;
    }

    // Did not find a chunk that fits the allocation

    RgBufferChunk *new_chunk = rgBufferChunkCreate(pool, allocate_size);
    if (last_chunk)
    {
        last_chunk->next = new_chunk;
    }
    else
    {
        pool->base_chunk = new_chunk;
    }

    chunk = new_chunk;
    goto buffer_pool_allocate_use_block;

    return alloc;
}

static void rgBufferPoolDestroy(RgBufferPool *pool)
{
    rgBufferChunkDestroy(pool->base_chunk);
}
// }}}

// Descriptor set allocator {{{
static void rgDescriptorPoolInit(
    RgDevice *device,
    RgDescriptorPool *pool,
    uint32_t num_bindings,
    VkDescriptorSetLayoutBinding *bindings)
{
    memset(pool, 0, sizeof(*pool));

    pool->device = device;
    pool->num_bindings = num_bindings;

    // Create set layout
    {
        VkDescriptorSetLayoutCreateInfo create_info;
        memset(&create_info, 0, sizeof(create_info));
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create_info.bindingCount = num_bindings;
        create_info.pBindings = bindings;

        VK_CHECK(vkCreateDescriptorSetLayout(
            device->device, &create_info, NULL, &pool->set_layout));
    }

    // Create update template
    {
        VkDescriptorUpdateTemplateEntry entries[RG_MAX_DESCRIPTOR_BINDINGS];

        for (uint32_t b = 0; b < num_bindings; ++b)
        {
            VkDescriptorSetLayoutBinding *binding = &bindings[b];
            assert(b == binding->binding);

            VkDescriptorUpdateTemplateEntry entry;
            memset(&entry, 0, sizeof(entry));
            entry.dstBinding = binding->binding;
            entry.dstArrayElement = 0;
            entry.descriptorCount = binding->descriptorCount;
            entry.descriptorType = binding->descriptorType;
            entry.offset = binding->binding * sizeof(RgDescriptor);
            entry.stride = sizeof(RgDescriptor);

            entries[b] = entry;
        }

        VkDescriptorUpdateTemplateCreateInfo template_info;
        memset(&template_info, 0, sizeof(template_info));
        template_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
        template_info.descriptorUpdateEntryCount = num_bindings;
        template_info.pDescriptorUpdateEntries = entries;
        template_info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
        template_info.descriptorSetLayout = pool->set_layout;

        VK_CHECK(vkCreateDescriptorUpdateTemplate(
            device->device, &template_info, NULL, &pool->update_template));
    }

    {
        for (uint32_t b = 0; b < num_bindings; ++b)
        {
            VkDescriptorSetLayoutBinding *binding = &bindings[b];

            VkDescriptorPoolSize *found_pool_size = NULL;

            for (uint32_t p = 0; p < pool->num_pool_sizes; ++p)
            {
                VkDescriptorPoolSize *pool_size = &pool->pool_sizes[p];
                if (pool_size->type == binding->descriptorType)
                {
                    found_pool_size = pool_size;
                    break;
                }
            }

            if (!found_pool_size)
            {
                VkDescriptorPoolSize new_pool_size = {binding->descriptorType, 0};
                pool->pool_sizes[pool->num_pool_sizes++] = new_pool_size;
                found_pool_size = &pool->pool_sizes[pool->num_pool_sizes - 1];
            }

            found_pool_size->descriptorCount += RG_SETS_PER_PAGE;
        }

        assert(pool->num_pool_sizes > 0);
    }
}

static void rgDescriptorPoolDestroy(RgDescriptorPool *pool)
{
    RgDescriptorPoolChunk *chunk = pool->base_chunk;
    while (chunk)
    {
        vkDestroyDescriptorPool(pool->device->device, chunk->pool, NULL);
        rgHashmapDestroy(&chunk->map);

        RgDescriptorPoolChunk *last_chunk = chunk;
        chunk = chunk->next;
        free(last_chunk);
    }

    vkDestroyDescriptorSetLayout(pool->device->device, pool->set_layout, NULL);
    vkDestroyDescriptorUpdateTemplate(pool->device->device, pool->update_template, NULL);
}

static void rgDescriptorPoolGrow(RgDescriptorPool *pool)
{
    RgDescriptorPoolChunk *chunk = (RgDescriptorPoolChunk *)malloc(sizeof(*chunk));
    memset(chunk, 0, sizeof(*chunk));

    chunk->next = pool->base_chunk;
    pool->base_chunk = chunk;

    rgHashmapInit(&chunk->map, RG_SETS_PER_PAGE);

    VkDescriptorPoolCreateInfo pool_create_info;
    memset(&pool_create_info, 0, sizeof(pool_create_info));
    pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_create_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_create_info.maxSets = RG_SETS_PER_PAGE;
    pool_create_info.poolSizeCount = pool->num_pool_sizes;
    pool_create_info.pPoolSizes = pool->pool_sizes;

    VK_CHECK(vkCreateDescriptorPool(
        pool->device->device, &pool_create_info, NULL, &chunk->pool));

    // Allocate descriptor sets
    VkDescriptorSetLayout set_layouts[RG_SETS_PER_PAGE];
    for (uint32_t i = 0; i < RG_SETS_PER_PAGE; i++)
    {
        set_layouts[i] = pool->set_layout;
    }

    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = chunk->pool;
    alloc_info.descriptorSetCount = RG_SETS_PER_PAGE;
    alloc_info.pSetLayouts = set_layouts;

    VK_CHECK(vkAllocateDescriptorSets(pool->device->device, &alloc_info, chunk->sets));
}

static VkDescriptorSet rgDescriptorPoolAllocate(
    RgDescriptorPool *pool, uint32_t num_descriptors, RgDescriptor *descriptors)
{
    assert(num_descriptors == pool->num_bindings);

    uint64_t descriptors_hash = 0;
    fnvHashReset(&descriptors_hash);
    fnvHashUpdate(
        &descriptors_hash,
        (uint8_t *)descriptors,
        sizeof(RgDescriptor) * num_descriptors);

    RgDescriptorPoolChunk *chunk = pool->base_chunk;
    while (chunk)
    {
        uint64_t *set_index_ptr = rgHashmapGet(&chunk->map, descriptors_hash);

        if (set_index_ptr != NULL)
        {
            // Set is available
            return chunk->sets[*set_index_ptr];
        }
        else
        {
            if (chunk->allocated_count >= RG_SETS_PER_PAGE)
            {
                // No sets available in this pool, so continue looking for another one
                continue;
            }

            // Update existing descriptor set, because we haven't found any
            // matching ones already

            uint64_t set_index = chunk->allocated_count;
            chunk->allocated_count++;

            VkDescriptorSet set = chunk->sets[set_index];
            rgHashmapSet(&chunk->map, descriptors_hash, set_index);

            vkUpdateDescriptorSetWithTemplate(
                pool->device->device, set, pool->update_template, descriptors);
            return set;
        }

        chunk = chunk->next;
    }

    rgDescriptorPoolGrow(pool);
    return rgDescriptorPoolAllocate(pool, num_descriptors, descriptors);
}
// }}}

// Pipeline {{{
RgPipeline *rgPipelineCreate(RgDevice *device, RgPipelineInfo *info)
{
    RgPipeline *pipeline = (RgPipeline *)malloc(sizeof(RgPipeline));
    memset(pipeline, 0, sizeof(*pipeline));

    pipeline->info = *info;
    pipeline->info.bindings =
        malloc(pipeline->info.num_bindings * sizeof(*pipeline->info.bindings));
    memcpy(
        pipeline->info.bindings,
        info->bindings,
        pipeline->info.num_bindings * sizeof(*pipeline->info.bindings));
    pipeline->info.vertex_attributes = malloc(
        pipeline->info.num_vertex_attributes * sizeof(*pipeline->info.vertex_attributes));
    memcpy(
        pipeline->info.vertex_attributes,
        info->vertex_attributes,
        pipeline->info.num_vertex_attributes * sizeof(*pipeline->info.vertex_attributes));

    pipeline->info.vertex = NULL;
    pipeline->info.vertex_size = 0;
    pipeline->info.fragment = NULL;
    pipeline->info.fragment_size = 0;

    rgHashmapInit(&pipeline->instances, 8);

    //
    // Create descriptor pools
    //

    VkDescriptorSetLayoutBinding bindings[RG_MAX_DESCRIPTOR_SETS]
                                         [RG_MAX_DESCRIPTOR_BINDINGS];
    uint32_t binding_counts[RG_MAX_DESCRIPTOR_SETS] = {0};
    uint32_t num_sets = 0;

    for (uint32_t i = 0; i < info->num_bindings; ++i)
    {
        RgPipelineBinding *binding = &info->bindings[i];
        VkDescriptorSetLayoutBinding *vk_binding =
            &bindings[binding->set][binding->binding];
        memset(vk_binding, 0, sizeof(*vk_binding));

        num_sets = RG_MAX(num_sets, binding->set + 1);
        binding_counts[binding->set] =
            RG_MAX(binding_counts[binding->set], binding->binding + 1);

        vk_binding->binding = binding->binding;
        vk_binding->descriptorType = pipeline_binding_type_to_vk(binding->type);
        vk_binding->descriptorCount = 1;
        vk_binding->stageFlags = VK_SHADER_STAGE_ALL; // TODO: this could be more specific
    }

    pipeline->num_sets = num_sets;

    VkDescriptorSetLayout set_layouts[RG_MAX_DESCRIPTOR_SETS];
    for (uint32_t i = 0; i < pipeline->num_sets; ++i)
    {
        rgDescriptorPoolInit(device, &pipeline->pools[i], binding_counts[i], bindings[i]);
        set_layouts[i] = pipeline->pools[i].set_layout;
    }

    //
    // Create pipeline layout
    //

    VkPipelineLayoutCreateInfo pipeline_layout_info;
    memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = pipeline->num_sets;
    pipeline_layout_info.pSetLayouts = set_layouts;
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = NULL;

    VK_CHECK(vkCreatePipelineLayout(
        device->device, &pipeline_layout_info, NULL, &pipeline->pipeline_layout));

    if (info->vertex && info->vertex_size > 0)
    {
        VkShaderModuleCreateInfo module_create_info;
        memset(&module_create_info, 0, sizeof(module_create_info));
        module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_create_info.codeSize = info->vertex_size;
        module_create_info.pCode = (uint32_t *)info->vertex;

        VK_CHECK(vkCreateShaderModule(
            device->device, &module_create_info, NULL, &pipeline->vertex_shader));
    }

    if (info->fragment && info->fragment_size > 0)
    {
        VkShaderModuleCreateInfo module_create_info;
        memset(&module_create_info, 0, sizeof(module_create_info));
        module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_create_info.codeSize = info->fragment_size;
        module_create_info.pCode = (uint32_t *)info->fragment;

        VK_CHECK(vkCreateShaderModule(
            device->device, &module_create_info, NULL, &pipeline->fragment_shader));
    }

    return pipeline;
}

void rgPipelineDestroy(RgDevice *device, RgPipeline *pipeline)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));

    free(pipeline->info.bindings);
    free(pipeline->info.vertex_attributes);

    for (uint32_t i = 0; i < pipeline->num_sets; ++i)
    {
        rgDescriptorPoolDestroy(&pipeline->pools[i]);
    }

    for (uint32_t i = 0; i < pipeline->instances.size; ++i)
    {
        if (pipeline->instances.hashes[i] != 0)
        {
            VkPipeline instance = VK_NULL_HANDLE;
            memcpy(&instance, &pipeline->instances.values[i], sizeof(VkPipeline));
            assert(instance != VK_NULL_HANDLE);
            vkDestroyPipeline(device->device, instance, NULL);
        }
    }

    if (pipeline->vertex_shader)
    {
        vkDestroyShaderModule(device->device, pipeline->vertex_shader, NULL);
    }

    if (pipeline->fragment_shader)
    {
        vkDestroyShaderModule(device->device, pipeline->fragment_shader, NULL);
    }

    vkDestroyPipelineLayout(device->device, pipeline->pipeline_layout, NULL);

    rgHashmapDestroy(&pipeline->instances);
    free(pipeline);
}

static VkPipeline
rgPipelineGetInstance(RgDevice *device, RgPipeline *pipeline, RgPass *pass)
{
    uint64_t *found = rgHashmapGet(&pipeline->instances, pass->hash);
    if (found)
    {
        VkPipeline instance;
        memcpy(&instance, found, sizeof(VkPipeline));
        return instance;
    }

    uint32_t num_stages = 0;
    VkPipelineShaderStageCreateInfo stages[RG_MAX_SHADER_STAGES];
    memset(stages, 0, sizeof(stages));

    if (pipeline->vertex_shader)
    {
        VkPipelineShaderStageCreateInfo *stage = &stages[num_stages++];
        stage->sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage->stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage->module = pipeline->vertex_shader;
        stage->pName = pipeline->info.vertex_entry ? pipeline->info.vertex_entry : "main";
    }

    if (pipeline->fragment_shader)
    {
        VkPipelineShaderStageCreateInfo *stage = &stages[num_stages++];
        stage->sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage->stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage->module = pipeline->fragment_shader;
        stage->pName =
            pipeline->info.fragment_entry ? pipeline->info.fragment_entry : "main";
    }

    VkPipelineVertexInputStateCreateInfo vertex_input_info;
    memset(&vertex_input_info, 0, sizeof(vertex_input_info));
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkVertexInputBindingDescription vertex_binding;
    memset(&vertex_binding, 0, sizeof(vertex_binding));

    VkVertexInputAttributeDescription attributes[RG_MAX_VERTEX_ATTRIBUTES];
    memset(attributes, 0, sizeof(attributes));

    if (pipeline->info.vertex_stride > 0)
    {
        assert(pipeline->info.num_vertex_attributes > 0);

        vertex_binding.binding = 0;
        vertex_binding.stride = pipeline->info.vertex_stride;

        vertex_input_info.vertexBindingDescriptionCount = 1;
        vertex_input_info.pVertexBindingDescriptions = &vertex_binding;

        for (uint32_t i = 0; i < pipeline->info.num_vertex_attributes; ++i)
        {
            attributes[i].binding = 0;
            attributes[i].location = i;
            attributes[i].format =
                format_to_vk(pipeline->info.vertex_attributes[i].format);
            attributes[i].offset = pipeline->info.vertex_attributes[i].offset;
        }

        vertex_input_info.vertexAttributeDescriptionCount =
            pipeline->info.num_vertex_attributes;
        vertex_input_info.pVertexAttributeDescriptions = attributes;
    }

    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    memset(&input_assembly, 0, sizeof(input_assembly));
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = primitive_topology_to_vk(pipeline->info.topology);
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport;
    memset(&viewport, 0, sizeof(viewport));
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)pass->extent.width;
    viewport.height = (float)pass->extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor;
    memset(&scissor, 0, sizeof(scissor));
    scissor.offset = (VkOffset2D){0, 0};
    scissor.extent = pass->extent;

    VkPipelineViewportStateCreateInfo viewport_state;
    memset(&viewport_state, 0, sizeof(viewport_state));
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer;
    memset(&rasterizer, 0, sizeof(rasterizer));
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = polygon_mode_to_vk(pipeline->info.polygon_mode);
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = cull_mode_to_vk(pipeline->info.cull_mode);
    rasterizer.frontFace = front_face_to_vk(pipeline->info.front_face);
    rasterizer.depthBiasEnable = pipeline->info.depth_stencil.bias_enable;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling;
    memset(&multisampling, 0, sizeof(multisampling));
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 0.0f;          // Optional
    multisampling.pSampleMask = NULL;               // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE;      // Optional

    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    memset(&depth_stencil, 0, sizeof(depth_stencil));
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = pipeline->info.depth_stencil.test_enable;
    depth_stencil.depthWriteEnable = pipeline->info.depth_stencil.write_enable;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState color_blend_attachment_enabled;
    memset(&color_blend_attachment_enabled, 0, sizeof(color_blend_attachment_enabled));
    color_blend_attachment_enabled.blendEnable = VK_TRUE;
    color_blend_attachment_enabled.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment_enabled.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_enabled.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_enabled.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_enabled.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment_enabled.alphaBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_enabled.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendAttachmentState color_blend_attachment_disabled;
    memset(&color_blend_attachment_disabled, 0, sizeof(color_blend_attachment_disabled));
    color_blend_attachment_disabled.blendEnable = VK_FALSE;
    color_blend_attachment_disabled.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment_disabled.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_disabled.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_disabled.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_disabled.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment_disabled.alphaBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_disabled.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendAttachmentState blend_infos[RG_MAX_COLOR_ATTACHMENTS];
    assert(pass->num_color_attachments <= RG_LENGTH(blend_infos));

    if (pipeline->info.blend.enable)
    {
        for (uint32_t i = 0; i < RG_LENGTH(blend_infos); ++i)
        {
            blend_infos[i] = color_blend_attachment_enabled;
        }
    }
    else
    {
        for (uint32_t i = 0; i < RG_LENGTH(blend_infos); ++i)
        {
            blend_infos[i] = color_blend_attachment_disabled;
        }
    }

    VkPipelineColorBlendStateCreateInfo color_blending;
    memset(&color_blending, 0, sizeof(color_blending));
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY; // Optional
    color_blending.attachmentCount = pass->num_color_attachments;
    color_blending.pAttachments = blend_infos;

    VkDynamicState dynamic_states[3] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state;
    memset(&dynamic_state, 0, sizeof(dynamic_state));
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = RG_LENGTH(dynamic_states);
    dynamic_state.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = num_stages;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = pipeline->pipeline_layout;
    pipeline_info.renderPass = pass->renderpass;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;

    VkPipeline instance = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(
        device->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &instance));

    uint64_t instance_id = 0;
    memcpy(&instance_id, &instance, sizeof(VkPipeline));
    rgHashmapSet(&pipeline->instances, pass->hash, instance_id);

    return instance;
}
// }}}

// Command buffer {{{
static void allocateCmdBuffer(RgDevice *device, RgCmdBuffer *cmd_buffer)
{
    memset(cmd_buffer, 0, sizeof(*cmd_buffer));

    cmd_buffer->device = device;

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = device->graphics_command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK(
        vkAllocateCommandBuffers(device->device, &alloc_info, &cmd_buffer->cmd_buffer));

    rgBufferPoolInit(
        device,
        &cmd_buffer->ubo_pool,
        RG_BUFFER_POOL_CHUNK_SIZE, /*chunk size*/
        RG_MAX(
            16,
            device->physical_device_properties.limits
                .minUniformBufferOffsetAlignment), /*alignment*/
        RG_BUFFER_USAGE_UNIFORM);

    rgBufferPoolInit(
        device,
        &cmd_buffer->vbo_pool,
        RG_BUFFER_POOL_CHUNK_SIZE, /*chunk size*/
        16,                        /*alignment*/
        RG_BUFFER_USAGE_VERTEX);

    rgBufferPoolInit(
        device,
        &cmd_buffer->ibo_pool,
        RG_BUFFER_POOL_CHUNK_SIZE, /*chunk size*/
        16,                        /*alignment*/
        RG_BUFFER_USAGE_INDEX);
}

static void freeCmdBuffer(RgDevice *device, RgCmdBuffer *cmd_buffer)
{
    rgBufferPoolDestroy(&cmd_buffer->ubo_pool);
    rgBufferPoolDestroy(&cmd_buffer->vbo_pool);
    rgBufferPoolDestroy(&cmd_buffer->ibo_pool);

    vkFreeCommandBuffers(
        device->device, device->graphics_command_pool, 1, &cmd_buffer->cmd_buffer);
}

static void cmdBufferBindDescriptors(RgCmdBuffer *cmd_buffer)
{
    assert(cmd_buffer->current_pipeline);

    for (uint32_t i = 0; i < cmd_buffer->current_pipeline->num_sets; ++i)
    {
        RgDescriptorPool *pool = &cmd_buffer->current_pipeline->pools[i];

        RgDescriptor *descriptors = &cmd_buffer->bound_descriptors[i][0];

        VkDescriptorSet descriptor_set =
            rgDescriptorPoolAllocate(pool, pool->num_bindings, descriptors);

        vkCmdBindDescriptorSets(
            cmd_buffer->cmd_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            cmd_buffer->current_pipeline->pipeline_layout,
            0,
            1,
            &descriptor_set,
            0,
            NULL);
    }
}

void rgCmdBindPipeline(RgCmdBuffer *cmd_buffer, RgPipeline *pipeline)
{
    RgPass *pass = cmd_buffer->current_pass;
    assert(pass);

    cmd_buffer->current_pipeline = pipeline;

    VkPipeline instance = rgPipelineGetInstance(cmd_buffer->device, pipeline, pass);

    assert(instance != VK_NULL_HANDLE);

    vkCmdBindPipeline(cmd_buffer->cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, instance);
}

void rgCmdBindImage(
    RgCmdBuffer *cmd_buffer, uint32_t binding, uint32_t set, RgImage *image)
{
    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.image.imageView = image->view;
    descriptor.image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    cmd_buffer->bound_descriptors[set][binding] = descriptor;
}

void rgCmdBindSampler(
    RgCmdBuffer *cmd_buffer, uint32_t binding, uint32_t set, RgSampler *sampler)
{
    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.image.sampler = sampler->sampler;

    cmd_buffer->bound_descriptors[set][binding] = descriptor;
}

void rgCmdBindImageSampler(
    RgCmdBuffer *cmd_buffer,
    uint32_t binding,
    uint32_t set,
    RgImage *image,
    RgSampler *sampler)
{
    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.image.imageView = image->view;
    descriptor.image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    descriptor.image.sampler = sampler->sampler;

    cmd_buffer->bound_descriptors[set][binding] = descriptor;
}

void rgCmdSetUniform(
    RgCmdBuffer *cmd_buffer, uint32_t binding, uint32_t set, size_t size, void *data)
{
    RgBufferAllocation alloc = rgBufferPoolAllocate(&cmd_buffer->ubo_pool, size);
    memcpy(alloc.mapping, data, size);

    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.buffer.buffer = alloc.buffer->buffer;
    descriptor.buffer.offset = alloc.offset;
    descriptor.buffer.range = alloc.size;

    cmd_buffer->bound_descriptors[set][binding] = descriptor;
}

void rgCmdSetVertices(RgCmdBuffer *cmd_buffer, size_t size, void *data)
{
    RgBufferAllocation alloc = rgBufferPoolAllocate(&cmd_buffer->vbo_pool, size);
    memcpy(alloc.mapping, data, size);

    vkCmdBindVertexBuffers(
        cmd_buffer->cmd_buffer, 0, 1, &alloc.buffer->buffer, &alloc.offset);
}

void rgCmdSetIndices(
    RgCmdBuffer *cmd_buffer, RgIndexType index_type, size_t size, void *data)
{
    RgBufferAllocation alloc = rgBufferPoolAllocate(&cmd_buffer->ibo_pool, size);
    memcpy(alloc.mapping, data, size);

    vkCmdBindIndexBuffer(
        cmd_buffer->cmd_buffer,
        alloc.buffer->buffer,
        alloc.offset,
        index_type_to_vk(index_type));
}

void rgCmdBindVertexBuffer(RgCmdBuffer *cmd_buffer, RgBuffer *buffer, size_t offset)
{
    vkCmdBindVertexBuffers(cmd_buffer->cmd_buffer, 0, 1, &buffer->buffer, &offset);
}

void rgCmdBindIndexBuffer(
    RgCmdBuffer *cmd_buffer, RgIndexType index_type, RgBuffer *buffer, size_t offset)
{
    vkCmdBindIndexBuffer(
        cmd_buffer->cmd_buffer, buffer->buffer, offset, index_type_to_vk(index_type));
}

void rgCmdDraw(
    RgCmdBuffer *cmd_buffer,
    uint32_t vertex_count,
    uint32_t instance_count,
    uint32_t first_vertex,
    uint32_t first_instance)
{
    cmdBufferBindDescriptors(cmd_buffer);

    vkCmdDraw(
        cmd_buffer->cmd_buffer,
        vertex_count,
        instance_count,
        first_vertex,
        first_instance);
}

void rgCmdDrawIndexed(
    RgCmdBuffer *cmd_buffer,
    uint32_t index_count,
    uint32_t instance_count,
    uint32_t first_index,
    int32_t vertex_offset,
    uint32_t first_instance)
{
    cmdBufferBindDescriptors(cmd_buffer);

    vkCmdDrawIndexed(
        cmd_buffer->cmd_buffer,
        index_count,
        instance_count,
        first_index,
        vertex_offset,
        first_instance);
}

void rgCmdDispatch(
    RgCmdBuffer *cmd_buffer,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z)
{
    cmdBufferBindDescriptors(cmd_buffer);
    vkCmdDispatch(cmd_buffer->cmd_buffer, group_count_x, group_count_y, group_count_z);
}

void rgCmdCopyBufferToBuffer(
    RgCmdBuffer *cmd_buffer,
    RgBuffer *src,
    size_t src_offset,
    RgBuffer *dst,
    size_t dst_offset,
    size_t size)
{
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size;
    vkCmdCopyBuffer(cmd_buffer->cmd_buffer, src->buffer, dst->buffer, 1, &region);
}

void rgCmdCopyBufferToImage(
    RgCmdBuffer *cmd_buffer, RgBufferCopy *src, RgImageCopy *dst, RgExtent3D extent)
{
    VkImageSubresourceLayers subresource;
    memset(&subresource, 0, sizeof(subresource));
    subresource.aspectMask = dst->image->aspect;
    subresource.mipLevel = dst->mip_level;
    subresource.baseArrayLayer = dst->array_layer;
    subresource.layerCount = 1;

    VkBufferImageCopy region = {
        .bufferOffset = src->offset,
        .bufferRowLength = src->row_length,
        .bufferImageHeight = src->image_height,
        .imageSubresource = subresource,
        .imageOffset =
            (VkOffset3D){
                .x = dst->offset.x,
                .y = dst->offset.y,
                .z = dst->offset.z,
            },
        .imageExtent =
            (VkExtent3D){
                .width = extent.width,
                .height = extent.height,
                .depth = extent.depth,
            },
    };

    vkCmdCopyBufferToImage(
        cmd_buffer->cmd_buffer,
        src->buffer->buffer,
        dst->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);
}

void rgCmdCopyImageToBuffer(
    RgCmdBuffer *cmd_buffer, RgImageCopy *src, RgBufferCopy *dst, RgExtent3D extent)
{
    VkImageSubresourceLayers subresource;
    memset(&subresource, 0, sizeof(subresource));
    subresource.aspectMask = src->image->aspect;
    subresource.mipLevel = src->mip_level;
    subresource.baseArrayLayer = src->array_layer;
    subresource.layerCount = 1;

    VkBufferImageCopy region = {
        .bufferOffset = dst->offset,
        .bufferRowLength = dst->row_length,
        .bufferImageHeight = dst->image_height,
        .imageSubresource = subresource,
        .imageOffset =
            (VkOffset3D){
                .x = src->offset.x,
                .y = src->offset.y,
                .z = src->offset.z,
            },
        .imageExtent =
            (VkExtent3D){
                .width = extent.width,
                .height = extent.height,
                .depth = extent.depth,
            },
    };

    vkCmdCopyImageToBuffer(
        cmd_buffer->cmd_buffer,
        src->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst->buffer->buffer,
        1,
        &region);
}

void rgCmdCopyImageToImage(
    RgCmdBuffer *cmd_buffer, RgImageCopy *src, RgImageCopy *dst, RgExtent3D extent)
{
    VkImageSubresourceLayers src_subresource;
    memset(&src_subresource, 0, sizeof(src_subresource));
    src_subresource.aspectMask = src->image->aspect;
    src_subresource.mipLevel = src->mip_level;
    src_subresource.baseArrayLayer = src->array_layer;
    src_subresource.layerCount = 1;

    VkImageSubresourceLayers dst_subresource;
    memset(&dst_subresource, 0, sizeof(dst_subresource));
    dst_subresource.aspectMask = dst->image->aspect;
    dst_subresource.mipLevel = dst->mip_level;
    dst_subresource.baseArrayLayer = dst->array_layer;
    dst_subresource.layerCount = 1;

    VkImageCopy region = {
        .srcSubresource = src_subresource,
        .srcOffset = {.x = src->offset.x, .y = src->offset.y, .z = src->offset.z},
        .dstSubresource = dst_subresource,
        .dstOffset = {.x = dst->offset.x, .y = dst->offset.y, .z = dst->offset.z},
        .extent = {.width = extent.width, .height = extent.height, .depth = extent.depth},
    };

    vkCmdCopyImage(
        cmd_buffer->cmd_buffer,
        src->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);
}
// }}}

// Graph {{{
static void
rgNodeInit(RgGraph *graph, RgNode *node, uint32_t *pass_indices, uint32_t pass_count)
{
    memset(node, 0, sizeof(*node));

    bool is_last = node == &graph->nodes[graph->num_nodes - 1];

    node->num_pass_indices = pass_count;
    node->pass_indices = (uint32_t *)malloc(sizeof(*node->pass_indices) * pass_count);
    memcpy(node->pass_indices, pass_indices, sizeof(*node->pass_indices) * pass_count);

    for (uint32_t i = 0; i < RG_FRAMES_IN_FLIGHT; ++i)
    {
        allocateCmdBuffer(graph->device, &node->frames[i].cmd_buffer);

        VkSemaphoreCreateInfo semaphore_info;
        memset(&semaphore_info, 0, sizeof(semaphore_info));
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(
            graph->device->device,
            &semaphore_info,
            NULL,
            &node->frames[i].execution_finished_semaphore));

        VkFenceCreateInfo fence_info;
        memset(&fence_info, 0, sizeof(fence_info));
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(
            graph->device->device, &fence_info, NULL, &node->frames[i].fence));

        uint32_t num_wait_semaphores = 0;
        VkSemaphore *wait_semaphores = NULL;
        VkPipelineStageFlags *wait_stages = NULL;

        if (is_last)
        {
            // Last node has to wait for the swapchain image to be available
            num_wait_semaphores++;
        }

        wait_semaphores =
            (VkSemaphore *)malloc(sizeof(*wait_semaphores) * num_wait_semaphores);
        wait_stages =
            (VkPipelineStageFlags *)malloc(sizeof(*wait_stages) * num_wait_semaphores);

        if (is_last)
        {
            wait_semaphores[num_wait_semaphores - 1] =
                graph->image_available_semaphores[i];
            wait_stages[num_wait_semaphores - 1] =
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }

        node->frames[i].num_wait_semaphores = num_wait_semaphores;
        node->frames[i].wait_semaphores = wait_semaphores;
        node->frames[i].wait_stages = wait_stages;
    }
}

static void rgNodeDestroy(RgDevice *device, RgNode *node)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));

    for (uint32_t i = 0; i < RG_FRAMES_IN_FLIGHT; ++i)
    {
        vkDestroySemaphore(
            device->device, node->frames[i].execution_finished_semaphore, NULL);

        vkDestroyFence(device->device, node->frames[i].fence, NULL);

        freeCmdBuffer(device, &node->frames[i].cmd_buffer);
    }

    free(node->pass_indices);
}

static uint64_t rgRenderpassHash(VkRenderPassCreateInfo *ci)
{
    uint64_t hash = 0;
    fnvHashReset(&hash);

    fnvHashUpdate(
        &hash,
        (uint8_t *)ci->pAttachments,
        ci->attachmentCount * sizeof(*ci->pAttachments));

    for (uint32_t i = 0; i < ci->subpassCount; i++)
    {
        const VkSubpassDescription *subpass = &ci->pSubpasses[i];
        fnvHashUpdate(
            &hash,
            (uint8_t *)&subpass->pipelineBindPoint,
            sizeof(subpass->pipelineBindPoint));

        if (subpass->pColorAttachments)
        {
            fnvHashUpdate(
                &hash,
                (uint8_t *)subpass->pColorAttachments,
                subpass->colorAttachmentCount * sizeof(*subpass->pColorAttachments));
        }
    }

    fnvHashUpdate(
        &hash,
        (uint8_t *)ci->pDependencies,
        ci->dependencyCount * sizeof(*ci->pDependencies));

    return hash;
}

static void rgPassResize(RgGraph *graph, RgPass *pass)
{
    for (uint32_t i = 0; i < pass->num_framebuffers; ++i)
    {
        if (pass->framebuffers[i])
        {
            vkDestroyFramebuffer(graph->device->device, pass->framebuffers[i], NULL);
            pass->framebuffers[i] = VK_NULL_HANDLE;
        }
    }

    if (pass->renderpass)
    {
        vkDestroyRenderPass(graph->device->device, pass->renderpass, NULL);
        pass->renderpass = VK_NULL_HANDLE;
    }

    if (pass->num_framebuffers < graph->swapchain.num_images)
    {
        pass->num_framebuffers = graph->swapchain.num_images;

        if (pass->framebuffers)
        {
            free(pass->framebuffers);
        }

        pass->framebuffers =
            (VkFramebuffer *)malloc(sizeof(VkFramebuffer) * pass->num_framebuffers);
        memset(pass->framebuffers, 0, sizeof(VkFramebuffer) * pass->num_framebuffers);
    }

    uint32_t num_rp_attachments = 0;
    VkAttachmentDescription rp_attachments[RG_MAX_ATTACHMENTS];
    memset(rp_attachments, 0, sizeof(rp_attachments));

    uint32_t num_color_attachment_refs = 0;
    VkAttachmentReference color_attachment_refs[RG_MAX_ATTACHMENTS];
    memset(color_attachment_refs, 0, sizeof(color_attachment_refs));

    bool has_depth_stencil = false;
    VkAttachmentReference depth_stencil_attachment_ref;
    memset(&depth_stencil_attachment_ref, 0, sizeof(depth_stencil_attachment_ref));

    if (graph->has_swapchain)
    {
        VkAttachmentDescription *backbuffer = &rp_attachments[num_rp_attachments++];
        backbuffer->format = graph->swapchain.image_format;
        backbuffer->samples = VK_SAMPLE_COUNT_1_BIT;
        backbuffer->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        backbuffer->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        backbuffer->stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        backbuffer->stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        backbuffer->initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        backbuffer->finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference *backbuffer_ref =
            &color_attachment_refs[num_color_attachment_refs++];
        backbuffer_ref->attachment = num_rp_attachments - 1;
        backbuffer_ref->layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    for (uint32_t i = 0; i < pass->num_outputs; ++i)
    {
        RgResource *resource = &graph->resources[pass->outputs[i]];

        switch (resource->type)
        {
        case RG_RESOURCE_COLOR_ATTACHMENT: {
            VkAttachmentDescription *attachment = &rp_attachments[num_rp_attachments++];
            attachment->format = format_to_vk(resource->image_info.format);
            attachment->samples =
                (VkSampleCountFlagBits)resource->image_info.sample_count;
            attachment->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment->stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment->stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment->initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachment->finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentReference *attachment_ref =
                &color_attachment_refs[num_color_attachment_refs++];
            attachment_ref->attachment = num_rp_attachments - 1;
            attachment_ref->layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            break;
        }
        case RG_RESOURCE_DEPTH_STENCIL_ATTACHMENT: {
            has_depth_stencil = true;

            VkAttachmentDescription *attachment = &rp_attachments[num_rp_attachments++];
            attachment->format = format_to_vk(resource->image_info.format);
            attachment->samples = VK_SAMPLE_COUNT_1_BIT;
            attachment->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment->stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment->stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment->initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachment->finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            pass->depth_attachment_index = num_rp_attachments - 1;
            depth_stencil_attachment_ref.attachment = num_rp_attachments - 1;
            depth_stencil_attachment_ref.layout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            break;
        }
        }
    }

    VkSubpassDescription subpass;
    memset(&subpass, 0, sizeof(subpass));
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = num_color_attachment_refs;
    subpass.pColorAttachments = color_attachment_refs;

    if (has_depth_stencil)
    {
        subpass.pDepthStencilAttachment = &depth_stencil_attachment_ref;
    }

    VkSubpassDependency dependency;
    memset(&dependency, 0, sizeof(dependency));
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderpass_ci;
    memset(&renderpass_ci, 0, sizeof(renderpass_ci));
    renderpass_ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderpass_ci.attachmentCount = num_rp_attachments;
    renderpass_ci.pAttachments = rp_attachments;
    renderpass_ci.subpassCount = 1;
    renderpass_ci.pSubpasses = &subpass;
    renderpass_ci.dependencyCount = 1;
    renderpass_ci.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(
        graph->device->device, &renderpass_ci, NULL, &pass->renderpass));

    pass->extent = graph->swapchain.extent;
    pass->hash = rgRenderpassHash(&renderpass_ci);

    for (uint32_t i = 0; i < pass->num_framebuffers; ++i)
    {
        uint32_t num_views = 0;
        VkImageView views[RG_MAX_ATTACHMENTS];
        memset(views, 0, sizeof(views));

        if (graph->has_swapchain)
        {
            views[num_views++] = graph->swapchain.image_views[i];
        }

        for (uint32_t i = 0; i < pass->num_outputs; ++i)
        {
            RgResource *resource = &graph->resources[pass->outputs[i]];

            switch (resource->type)
            {
            case RG_RESOURCE_COLOR_ATTACHMENT:
            case RG_RESOURCE_DEPTH_STENCIL_ATTACHMENT:
                views[num_views++] = resource->image->view;
                break;
            }
        }

        VkFramebufferCreateInfo create_info;
        memset(&create_info, 0, sizeof(create_info));
        create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        create_info.renderPass = pass->renderpass;
        create_info.attachmentCount = num_views;
        create_info.pAttachments = views;
        create_info.width = pass->extent.width;
        create_info.height = pass->extent.height;
        create_info.layers = 1;

        VK_CHECK(vkCreateFramebuffer(
            graph->device->device, &create_info, NULL, &pass->framebuffers[i]));
    }
}

static void rgPassBuild(RgGraph *graph, RgPass *pass)
{
    if (graph->has_swapchain)
    {
        pass->num_color_attachments++;
        pass->num_attachments++;
    }

    for (uint32_t i = 0; i < pass->num_outputs; ++i)
    {
        RgResource *resource = &graph->resources[pass->outputs[i]];
        switch (resource->type)
        {
        case RG_RESOURCE_COLOR_ATTACHMENT:
            pass->num_color_attachments++;
            pass->num_attachments++;
            break;
        case RG_RESOURCE_DEPTH_STENCIL_ATTACHMENT:
            pass->has_depth_attachment = true;
            pass->num_attachments++;
            break;
        default: break;
        }
    }
}

static void rgPassDestroy(RgDevice *device, RgPass *pass)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));
    if (pass->renderpass)
    {
        vkDestroyRenderPass(device->device, pass->renderpass, NULL);
        pass->renderpass = VK_NULL_HANDLE;
    }

    for (uint32_t i = 0; i < pass->num_framebuffers; ++i)
    {
        vkDestroyFramebuffer(device->device, pass->framebuffers[i], NULL);
        if (pass->framebuffers[i])
        {
            pass->framebuffers[i] = VK_NULL_HANDLE;
        }
    }

    if (pass->framebuffers)
    {
        free(pass->framebuffers);
    }
}

void rgGraphResize(RgGraph *graph)
{
    rgSwapchainResize(&graph->swapchain);

    for (uint32_t i = 0; i < graph->num_resources; ++i)
    {
        RgResource *resource = &graph->resources[i];
        switch (resource->type)
        {
        case RG_RESOURCE_COLOR_ATTACHMENT:
        case RG_RESOURCE_DEPTH_STENCIL_ATTACHMENT: {
            if (resource->image)
            {
                rgImageDestroy(graph->device, resource->image);
            }

            RgImageInfo image_info;
            memset(&image_info, 0, sizeof(image_info));

            image_info.depth = resource->image_info.depth;
            image_info.sample_count = resource->image_info.sample_count;
            image_info.mip_count = resource->image_info.mip_count;
            image_info.layer_count = resource->image_info.layer_count;
            image_info.usage = resource->image_info.usage;
            image_info.aspect = resource->image_info.aspect;
            image_info.format = resource->image_info.format;

            switch (resource->image_info.scaling_mode)
            {
            case RG_GRAPH_IMAGE_SCALING_MODE_ABSOLUTE: {
                image_info.width = (uint32_t)resource->image_info.width;
                image_info.height = (uint32_t)resource->image_info.height;
                break;
            }
            case RG_GRAPH_IMAGE_SCALING_MODE_RELATIVE: {
                assert(resource->image_info.width <= 1.0);
                assert(resource->image_info.height <= 1.0);

                assert(graph->has_swapchain);
                image_info.width = (uint32_t)(
                    resource->image_info.width * (float)graph->swapchain.extent.width);
                image_info.height = (uint32_t)(
                    resource->image_info.height * (float)graph->swapchain.extent.height);
                break;
            }
            }

            assert(image_info.width > 0);
            assert(image_info.height > 0);

            resource->image = rgImageCreate(graph->device, &image_info);
            break;
        }
        }
    }

    for (uint32_t i = 0; i < graph->num_passes; ++i)
    {
        rgPassResize(graph, &graph->passes[i]);
    }
}

RgGraph *rgGraphCreate(RgDevice *device, void *user_data, RgPlatformWindowInfo *window)
{
    RgGraph *graph = (RgGraph *)malloc(sizeof(RgGraph));
    memset(graph, 0, sizeof(*graph));

    graph->device = device;
    graph->user_data = user_data;

    if (window)
    {
        graph->has_swapchain = true;
        rgSwapchainInit(device, &graph->swapchain, window);
    }

    return graph;
}

RgPass *rgGraphAddPass(RgGraph *graph, RgPassCallback *callback)
{
    RgPass *pass = &graph->passes[graph->num_passes];
    assert(pass < (&graph->passes[RG_MAX_GRAPH_PASSES]));
    graph->num_passes++;

    memset(pass, 0, sizeof(*pass));

    pass->graph = graph;
    pass->callback = callback;

    return pass;
}

RgResource *rgGraphAddResource(RgGraph *graph, RgResourceInfo *info)
{
    RgResource *resource = &graph->resources[graph->num_resources];
    assert(resource < (&graph->resources[RG_MAX_GRAPH_RESOURCES]));
    graph->num_resources++;
    memset(resource, 0, sizeof(*resource));

    resource->type = info->type;
    switch (resource->type)
    {
    case RG_RESOURCE_DEPTH_STENCIL_ATTACHMENT:
        resource->image_info = info->image;
        resource->image_info.usage |= RG_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT;

        switch (resource->image_info.format)
        {
        case RG_FORMAT_D32_SFLOAT:
            resource->image_info.aspect |= RG_IMAGE_ASPECT_DEPTH;
            break;

        case RG_FORMAT_D24_UNORM_S8_UINT:
            resource->image_info.aspect |= RG_IMAGE_ASPECT_DEPTH;
            resource->image_info.aspect |= RG_IMAGE_ASPECT_STENCIL;
            break;

        default: assert(0);
        }
        break;

    case RG_RESOURCE_COLOR_ATTACHMENT:
        resource->image_info = info->image;
        resource->image_info.usage |= RG_IMAGE_USAGE_COLOR_ATTACHMENT;
        resource->image_info.aspect |= RG_IMAGE_ASPECT_COLOR;
        break;
    }

    return resource;
}

void rgGraphAddPassInput(RgPass *pass, RgResource *resource)
{
    uint32_t resource_index = (uint32_t)(resource - pass->graph->resources);

    uint32_t *resource_ptr = &pass->inputs[pass->num_inputs];
    assert(resource_ptr < (&pass->inputs[RG_MAX_PASS_RESOURCES]));

    *resource_ptr = resource_index;
    pass->num_inputs++;
}

void rgGraphAddPassOutput(RgPass *pass, RgResource *resource)
{
    uint32_t resource_index = (uint32_t)(resource - pass->graph->resources);

    uint32_t *resource_ptr = &pass->outputs[pass->num_outputs];
    assert(resource_ptr < (&pass->outputs[RG_MAX_PASS_RESOURCES]));

    *resource_ptr = resource_index;
    pass->num_outputs++;
}

void rgGraphBuild(RgGraph *graph)
{
    for (uint32_t i = 0; i < RG_FRAMES_IN_FLIGHT; ++i)
    {
        VkSemaphoreCreateInfo semaphore_info;
        memset(&semaphore_info, 0, sizeof(semaphore_info));
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(
            graph->device->device,
            &semaphore_info,
            NULL,
            &graph->image_available_semaphores[i]));
    }

    assert(graph->num_passes > 0);

    graph->num_nodes = 1;
    graph->nodes = (RgNode *)malloc(sizeof(*graph->nodes) * graph->num_nodes);
    for (uint32_t i = 0; i < graph->num_nodes; ++i)
    {
        uint32_t pass_indices[1] = {0};
        rgNodeInit(graph, &graph->nodes[i], pass_indices, 1);
    }

    for (uint32_t i = 0; i < graph->num_passes; ++i)
    {
        rgPassBuild(graph, &graph->passes[i]);
    }

    rgGraphResize(graph);

    graph->built = true;
}

void rgGraphDestroy(RgGraph *graph)
{
    VK_CHECK(vkDeviceWaitIdle(graph->device->device));

    for (uint32_t i = 0; i < RG_FRAMES_IN_FLIGHT; ++i)
    {
        vkDestroySemaphore(
            graph->device->device, graph->image_available_semaphores[i], NULL);
    }

    for (uint32_t i = 0; i < graph->num_nodes; ++i)
    {
        rgNodeDestroy(graph->device, &graph->nodes[i]);
    }
    free(graph->nodes);

    for (uint32_t i = 0; i < graph->num_passes; ++i)
    {
        rgPassDestroy(graph->device, &graph->passes[i]);
    }

    for (uint32_t i = 0; i < graph->num_resources; ++i)
    {
        switch (graph->resources[i].type)
        {
        case RG_RESOURCE_DEPTH_STENCIL_ATTACHMENT:
        case RG_RESOURCE_COLOR_ATTACHMENT:
            if (graph->resources[i].image)
            {
                rgImageDestroy(graph->device, graph->resources[i].image);
                graph->resources[i].image = NULL;
            }
            break;
        }
    }

    if (graph->has_swapchain)
    {
        rgSwapchainDestroy(graph->device, &graph->swapchain);
    }

    free(graph);
}

void rgGraphExecute(RgGraph *graph)
{
    assert(graph->built);
    uint32_t current_frame = graph->current_frame;

    for (uint32_t i = 0; i < graph->num_nodes; ++i)
    {
        RgNode *node = &graph->nodes[i];

        if (i == (graph->num_nodes - 1))
        {
            // Last node has to acquire swapchain image

            VK_CHECK(vkWaitForFences(
                graph->device->device,
                1,
                &node->frames[current_frame].fence,
                VK_TRUE,
                1 * 1000000000ULL));

            VkResult res;
            while (1)
            {
                res = vkAcquireNextImageKHR(
                    graph->device->device,
                    graph->swapchain.swapchain,
                    UINT64_MAX,
                    graph->image_available_semaphores[current_frame],
                    VK_NULL_HANDLE,
                    &graph->swapchain.current_image_index);

                if (res != VK_ERROR_OUT_OF_DATE_KHR)
                {
                    break;
                }

                rgSwapchainResize(&graph->swapchain);
            }

            VK_CHECK(res);
        }

        VK_CHECK(
            vkResetFences(graph->device->device, 1, &node->frames[current_frame].fence));

        uint32_t num_signal_semaphores = 1;
        VkSemaphore *signal_semaphores =
            &node->frames[current_frame].execution_finished_semaphore;

        RgCmdBuffer *cmd_buffer = &node->frames[current_frame].cmd_buffer;

        VkCommandBufferBeginInfo begin_info;
        memset(&begin_info, 0, sizeof(begin_info));
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_buffer->cmd_buffer, &begin_info));

        for (uint32_t j = 0; j < node->num_pass_indices; ++j)
        {
            RgPass *pass = &graph->passes[node->pass_indices[j]];

            pass->current_framebuffer =
                pass->framebuffers[graph->swapchain.current_image_index];

            uint32_t num_clear_values = pass->num_attachments;
            VkClearValue clear_values[RG_MAX_ATTACHMENTS];
            memset(clear_values, 0, sizeof(clear_values));

            if (pass->has_depth_attachment)
            {
                clear_values[pass->depth_attachment_index].depthStencil.depth = 1.0f;
                clear_values[pass->depth_attachment_index].depthStencil.stencil = 0;
            }

            VkRenderPassBeginInfo render_pass_info;
            memset(&render_pass_info, 0, sizeof(render_pass_info));
            render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            render_pass_info.renderPass = pass->renderpass;
            render_pass_info.framebuffer = pass->current_framebuffer;
            render_pass_info.renderArea.offset = (VkOffset2D){0, 0};
            render_pass_info.renderArea.extent = pass->extent;
            render_pass_info.clearValueCount = num_clear_values;
            render_pass_info.pClearValues = clear_values;

            cmd_buffer->current_pass = pass;
            vkCmdBeginRenderPass(
                cmd_buffer->cmd_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport;
            memset(&viewport, 0, sizeof(viewport));
            viewport.x = 0;
            viewport.y = 0;
            viewport.width = (float)pass->extent.width;
            viewport.height = (float)pass->extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd_buffer->cmd_buffer, 0, 1, &viewport);

            VkRect2D scissor;
            memset(&scissor, 0, sizeof(scissor));
            scissor.offset.x = 0;
            scissor.offset.y = 0;
            scissor.extent = pass->extent;

            vkCmdSetScissor(cmd_buffer->cmd_buffer, 0, 1, &scissor);

            assert(pass->callback);
            pass->callback(graph->user_data, cmd_buffer);

            vkCmdEndRenderPass(cmd_buffer->cmd_buffer);
            cmd_buffer->current_pass = NULL;
        }

        rgBufferPoolReset(&cmd_buffer->ubo_pool);
        rgBufferPoolReset(&cmd_buffer->vbo_pool);
        rgBufferPoolReset(&cmd_buffer->ibo_pool);

        VK_CHECK(vkEndCommandBuffer(cmd_buffer->cmd_buffer));

        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = node->frames[current_frame].num_wait_semaphores;
        submit.pWaitSemaphores = node->frames[current_frame].wait_semaphores;
        submit.pWaitDstStageMask = node->frames[current_frame].wait_stages;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd_buffer->cmd_buffer;
        submit.signalSemaphoreCount = num_signal_semaphores;
        submit.pSignalSemaphores = signal_semaphores;

        VK_CHECK(vkQueueSubmit(
            graph->device->graphics_queue,
            1,
            &submit,
            node->frames[current_frame].fence));
    }

    RgNode *last_node = &graph->nodes[graph->num_nodes - 1];

    VkPresentInfoKHR present_info;
    memset(&present_info, 0, sizeof(present_info));
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores =
        &last_node->frames[current_frame].execution_finished_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &graph->swapchain.swapchain;
    present_info.pImageIndices = &graph->swapchain.current_image_index;

    assert(graph->swapchain.swapchain != VK_NULL_HANDLE);

    VkResult res = vkQueuePresentKHR(graph->swapchain.present_queue, &present_info);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
    {
        rgSwapchainResize(&graph->swapchain);
    }
    else
    {
        VK_CHECK(res);
    }

    graph->current_frame = (graph->current_frame + 1) % RG_FRAMES_IN_FLIGHT;
}
// }}}

#endif // RENDERGRAPH_FEATURE_VULKAN
