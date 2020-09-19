#ifndef RENDERGRAPH_EXT_H

#include "rendergraph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RgExtCompiledShader
{
    uint8_t *code;
    size_t code_size;
    const char* entry_point;
} RgExtCompiledShader;

RgPipeline *rgExtPipelineCreateWithShaders(
        RgDevice *device,
        RgExtCompiledShader *vertex_shader,
        RgExtCompiledShader *fragment_shader,
        RgPipelineInfo *info);

#ifdef __cplusplus
}
#endif

#endif // RENDERGRAPH_EXT_H
