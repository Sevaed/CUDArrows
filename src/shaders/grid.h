#pragma once
#include "gl/uniform.h"
#include "shaders/shader.h"

const char *gridVertex = R"%==%(
#version 330 core
layout (location = 0) in vec4 position;

out vec2 texCoord;

uniform vec2 tileCount;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    texCoord = position.xy * tileCount;
    gl_Position = projection * view * model * position;
}
)%==%";

const char *gridFragment = R"%==%(
#version 330 core
#extension GL_OES_standard_derivatives : enable

out vec4 FragColor;

in vec2 texCoord;

float gridThickness = .06;

float filterWidth2(vec2 uv) {
    vec2 dx = dFdx(uv), dy = dFdy(uv);
    return dot(dx, dx) + dot(dy, dy) + .0001;
}

float gridSmooth(vec2 p) {
    vec2 q = p;
    q += .5;
    q -= floor(q);
    q = (gridThickness + 1.) * .5 - abs(q - .5);
    float w = 12. * filterWidth2(p);
    float s = sqrt(gridThickness);
    return smoothstep(.5 - w * s, .5 + w, max(q.x, q.y));
}

void main() {
    FragColor = vec4(0.0, 0.0, 0.0, 0.2) * gridSmooth(texCoord);
}
)%==%";

namespace cudarrows {
    class GridShader : public BaseShader {
    public:
        gl::Uniform2f tileCount;

        GridShader() : BaseShader(gridVertex, gridFragment),
            tileCount(program.getUniform<gl::Uniform2f>("tileCount")) {}
    };
};