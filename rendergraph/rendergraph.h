#ifndef RENDERGRAPH_H
#define RENDERGRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RENDERGRAPH_FEATURE_VULKAN
#define RENDERGRAPH_FEATURE_SDL2
#define RENDERGRAPH_FEATURE_VALIDATION

#ifdef RENDERGRAPH_FEATURE_SDL2
typedef struct SDL_Window SDL_Window;
#endif // RENDERGRAPH_FEATURE_SDL2

typedef struct RgDevice RgDevice;
typedef struct RgPipeline RgPipeline;
typedef struct RgBuffer RgBuffer;
typedef struct RgImage RgImage;
typedef struct RgSampler RgSampler;
typedef struct RgCmdBuffer RgCmdBuffer;
typedef struct RgGraph RgGraph;
typedef struct RgPass RgPass;
typedef struct RgNode RgNode;
typedef struct RgResource RgResource;
typedef uint32_t RgFlags;
typedef void(RgPassCallback)(void*, RgCmdBuffer*);

typedef enum
{
    RG_FORMAT_UNDEFINED = 0,

    RG_FORMAT_RGB8_UNORM,
    RG_FORMAT_RGBA8_UNORM,

    RG_FORMAT_R32_UINT,

    RG_FORMAT_R32_SFLOAT,
    RG_FORMAT_RG32_SFLOAT,
    RG_FORMAT_RGB32_SFLOAT,
    RG_FORMAT_RGBA32_SFLOAT,

    RG_FORMAT_RGBA16_SFLOAT,

    RG_FORMAT_D32_SFLOAT,
    RG_FORMAT_D24_UNORM_S8_UINT,
} RgFormat;

typedef enum
{
	RG_IMAGE_USAGE_SAMPLED                  = 1 << 0,
	RG_IMAGE_USAGE_TRANSFER_DST             = 1 << 1,
	RG_IMAGE_USAGE_TRANSFER_SRC             = 1 << 2,
	RG_IMAGE_USAGE_STORAGE                  = 1 << 3,
	RG_IMAGE_USAGE_COLOR_ATTACHMENT         = 1 << 4,
	RG_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT = 1 << 5,
} RgImageUsage;

typedef enum
{
	RG_IMAGE_ASPECT_COLOR   = 1 << 0,
	RG_IMAGE_ASPECT_DEPTH   = 1 << 1,
	RG_IMAGE_ASPECT_STENCIL = 1 << 2,
} RgImageAspect;

typedef enum
{
	RG_FILTER_LINEAR,
	RG_FILTER_NEAREST,
} RgFilter;

typedef enum
{
    RG_SAMPLER_ADDRESS_MODE_REPEAT,
    RG_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    RG_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
} RgSamplerAddressMode;

typedef enum
{
    RG_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    RG_BORDER_COLOR_INT_TRANSPARENT_BLACK,
    RG_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    RG_BORDER_COLOR_INT_OPAQUE_BLACK,
    RG_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    RG_BORDER_COLOR_INT_OPAQUE_WHITE,
} RgBorderColor;

typedef struct
{
	uint32_t width;
	uint32_t height;
	uint32_t depth;
	uint32_t sample_count;
	uint32_t mip_count;
	uint32_t layer_count;
	RgFlags usage;
	RgFlags aspect;
	RgFormat format;
} RgImageInfo;

typedef struct
{
    bool anisotropy;
    float min_lod;
    float max_lod;
    RgFilter mag_filter;
    RgFilter min_filter;
    RgSamplerAddressMode address_mode;
    RgBorderColor border_color;
} RgSamplerInfo;

typedef enum
{
	RG_BUFFER_USAGE_VERTEX       = 1 << 0,
	RG_BUFFER_USAGE_INDEX        = 1 << 1,
	RG_BUFFER_USAGE_UNIFORM      = 1 << 2,
	RG_BUFFER_USAGE_TRANSFER_SRC = 1 << 3,
	RG_BUFFER_USAGE_TRANSFER_DST = 1 << 4,
	RG_BUFFER_USAGE_STORAGE      = 1 << 5,
} RgBufferUsage;

typedef enum
{
	RG_BUFFER_MEMORY_HOST = 1,
	RG_BUFFER_MEMORY_DEVICE,
} RgBufferMemory;

typedef struct
{
	size_t size;
	RgFlags usage;
	RgBufferMemory memory;
} RgBufferInfo;

typedef enum
{
    RG_INDEX_TYPE_UINT32 = 0,
    RG_INDEX_TYPE_UINT16,
} RgIndexType;

typedef enum
{
    RG_POLYGON_MODE_FILL = 0,
    RG_POLYGON_MODE_LINE,
    RG_POLYGON_MODE_POINT,
} RgPolygonMode;

typedef enum
{
    RG_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 0,
    RG_PRIMITIVE_TOPOLOGY_LINE_LIST,
} RgPrimitiveTopology;

typedef enum
{
    RG_FRONT_FACE_CLOCKWISE = 0,
    RG_FRONT_FACE_COUNTER_CLOCKWISE,
} RgFrontFace;

typedef enum
{
    RG_CULL_MODE_NONE = 0,
    RG_CULL_MODE_BACK,
    RG_CULL_MODE_FRONT,
    RG_CULL_MODE_FRONT_AND_BACK,
} RgCullMode;

typedef enum
{
    RG_BINDING_UNIFORM_BUFFER = 1,
    RG_BINDING_STORAGE_BUFFER,
    RG_BINDING_IMAGE,
    RG_BINDING_SAMPLER,
    RG_BINDING_IMAGE_SAMPLER,
} RgPipelineBindingType;

typedef struct
{
    uint32_t set;
    uint32_t binding;
    RgPipelineBindingType type;
} RgPipelineBinding;

typedef struct
{
    RgFormat format;
    uint32_t offset;
} RgVertexAttribute;

typedef struct
{
    bool enable;
} RgPipelineBlendState;

typedef struct
{
    bool test_enable;
    bool write_enable;
    bool bias_enable;
} RgPipelineDepthStencilState;

typedef struct 
{
    RgPolygonMode       polygon_mode;
    RgCullMode          cull_mode;
    RgFrontFace         front_face;
    RgPrimitiveTopology topology;

    RgPipelineBlendState        blend;
    RgPipelineDepthStencilState depth_stencil;

    uint32_t vertex_stride;
    uint32_t num_vertex_attributes;
    RgVertexAttribute *vertex_attributes;

    uint32_t num_bindings;
    RgPipelineBinding *bindings;

    uint8_t *vertex;
    size_t vertex_size;
    const char* vertex_entry;

    uint8_t *fragment;
    size_t fragment_size;
    const char* fragment_entry;
} RgPipelineInfo;

typedef enum
{
    RG_RESOURCE_COLOR_ATTACHMENT,
    RG_RESOURCE_DEPTH_STENCIL_ATTACHMENT,
} RgResourceType;

typedef struct
{
    RgResourceType type;
    union 
    {
        RgImageInfo image;
        RgBufferInfo buffer;
    };
} RgResourceInfo;

typedef struct RgOffset3D
{
    int32_t x, y, z;
} RgOffset3D;

typedef struct RgExtent3D
{
    uint32_t width, height, depth;
} RgExtent3D;

typedef struct
{
    RgImage *image;
    uint32_t mip_level;
    uint32_t array_layer;
    RgOffset3D offset;
} RgImageCopy;

typedef struct {
    RgBuffer *buffer;
    size_t offset;
    uint32_t row_length;
    uint32_t image_height;
} RgBufferCopy;

RgDevice *rgDeviceCreate();
void rgDeviceDestroy(RgDevice* device);

RgImage *rgImageCreate(RgDevice *device, RgImageInfo *info);
void rgImageDestroy(RgDevice *device, RgImage *image);
void rgImageUpload(RgDevice *device, RgImageCopy *dst, RgExtent3D *extent, size_t size, void *data);

RgSampler *rgSamplerCreate(RgDevice *device, RgSamplerInfo *info);
void rgSamplerDestroy(RgDevice *device, RgSampler *sampler);

RgBuffer *rgBufferCreate(RgDevice *device, RgBufferInfo *info);
void rgBufferDestroy(RgDevice *device, RgBuffer *buffer);
void *rgBufferMap(RgDevice *device, RgBuffer *buffer);
void rgBufferUnmap(RgDevice *device, RgBuffer *buffer);
void rgBufferUpload(RgDevice *device, RgBuffer *buffer, size_t offset, size_t size, void *data);

RgPipeline *rgPipelineCreate(RgDevice *device, RgPipelineInfo *info);
void rgPipelineDestroy(RgDevice *device, RgPipeline *pipeline);

RgGraph *rgGraphCreate(RgDevice *device, void *user_data, void *window);
RgPass *rgGraphAddPass(RgGraph *graph, RgPassCallback *callback);
RgResource *rgGraphAddResource(RgGraph *graph, RgResourceInfo *info);
void rgGraphAddPassInput(RgPass *pass, RgResource *resource);
void rgGraphAddPassOutput(RgPass *pass, RgResource *resource);
void rgGraphBuild(RgGraph *graph);
void rgGraphDestroy(RgGraph *graph);
void rgGraphResize(RgGraph *graph);
void rgGraphExecute(RgGraph *graph);

void rgCmdBindPipeline(RgCmdBuffer *cb, RgPipeline *pipeline);
void rgCmdBindImage(RgCmdBuffer *cb, uint32_t binding, uint32_t set, RgImage *image);
void rgCmdBindSampler(RgCmdBuffer *cb, uint32_t binding, uint32_t set, RgSampler *sampler);
void rgCmdBindImageSampler(RgCmdBuffer *cb, uint32_t binding, uint32_t set, RgImage *image, RgSampler *sampler);
void rgCmdBindVertexBuffer(RgCmdBuffer *cb, RgBuffer *buffer, size_t offset);
void rgCmdBindIndexBuffer(RgCmdBuffer *cb, RgIndexType index_type, RgBuffer *buffer, size_t offset);
void rgCmdSetUniform(RgCmdBuffer *cb, uint32_t binding, uint32_t set, size_t size, void *data);
void rgCmdSetVertices(RgCmdBuffer *cb, size_t size, void *data);
void rgCmdSetIndices(RgCmdBuffer *cb, RgIndexType index_type, size_t size, void *data);
void rgCmdDraw(RgCmdBuffer *cb, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance);
void rgCmdDrawIndexed(RgCmdBuffer *cb, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance);
void rgCmdDispatch(RgCmdBuffer *cb, uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z);
void rgCmdCopyBufferToBuffer(RgCmdBuffer *cb, RgBuffer *src, size_t src_offset, RgBuffer *dst, size_t dst_offset, size_t size);
void rgCmdCopyBufferToImage(RgCmdBuffer *cb, RgBufferCopy *src, RgImageCopy *dst, RgExtent3D extent);
void rgCmdCopyImageToBuffer(RgCmdBuffer *cb, RgImageCopy *src, RgBufferCopy *dst, RgExtent3D extent);
void rgCmdCopyImageToImage(RgCmdBuffer *cb, RgImageCopy *src, RgImageCopy *dst, RgExtent3D extent);

#ifdef __cplusplus
}
#endif

#endif // RENDERGRAPH_H
