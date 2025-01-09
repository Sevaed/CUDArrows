#pragma once
#include "gl/uniform.h"
#include "shaders/shader.h"

const char *arrowsVertex = R"%==%(
#version 330 core
layout (location = 0) in vec4 position;

out vec2 texCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    texCoord = position.xy;
    gl_Position = projection * view * model * position;
}
)%==%";

const char *arrowsFragment = R"%==%(
#version 330 core

out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D arrowAtlas;
uniform sampler2D data;

const vec4 signalColors[] = vec4[](
    vec4(1.0, 1.0, 1.0, 0.0),
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(0.3, 0.5, 1.0, 1.0),
    vec4(1.0, 1.0, 0.0, 1.0),
    vec4(0.0, 0.8, 0.0, 1.0),
    vec4(1.0, 0.8, 0.2, 1.0),
    vec4(1.0, 0.2, 1.0, 1.0)
);

const float atlasSize = 8.0;

const vec2 uvCenter = vec2(0.5);

#define M_PI 3.1415926535897932384626433832795

vec2 rot(vec2 uv, float rotation, float flipped) {
    rotation = -rotation * M_PI / 2.0;
    flipped = 1.0 - flipped * 2.0;
    float s = sin(rotation);
    float c = cos(rotation);
    uv -= uvCenter;
    uv *= mat2(c, -s, s, c);
    uv.x *= flipped;
    uv += uvCenter;
    return uv;
}

void main() {
    vec2 texSize = vec2(textureSize(data, 0));
    vec4 arrowData = floor(texture2D(data, floor(texCoord * texSize) / texSize) * 255.0);
    float arrowType = arrowData.x;
    float arrowIndex = arrowType - 1.0;
    float rotation = arrowData.y;
    float flipped = floor(rotation / 4.0);
    rotation -= flipped * 4.0;
    int arrowSignal = int(arrowData.z);
    vec2 uv = fract(texCoord * texSize);
    uv = rot(uv, rotation, flipped);
    vec2 offset = vec2(mod(arrowIndex, atlasSize), floor(arrowIndex / atlasSize));
    vec4 texColor = texture2D(arrowAtlas, (offset + uv) / atlasSize);
    vec4 signalColor = signalColors[arrowSignal];
    FragColor = mix(signalColor, texColor, min(arrowType, 1.0) * texColor.a);
}
)%==%";

namespace cudarrows {
    class ArrowsShader : public BaseShader {
    public:
        gl::Uniform1i arrowAtlas;
        gl::Uniform1i data;

        ArrowsShader() : BaseShader(arrowsVertex, arrowsFragment),
            arrowAtlas(program.getUniform<gl::Uniform1i>("arrowAtlas")),
            data(program.getUniform<gl::Uniform1i>("data")) {}
    };
};