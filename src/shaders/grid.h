#pragma once
#include "gl/uniform.h"
#include "shaders/shader.h"

const char *gridVertex = R"%==%(
#version 330 core
layout (location = 0) in vec2 position;

out vec2 texCoord;

uniform mat4 transform;

void main() {
    vec4 texPos = vec4(position.x, 1.0 - position.y, 0.0, 1.0);
    texCoord = (transform * texPos).xy;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)%==%";

const char *gridFragment = R"%==%(
#version 330 core
#extension GL_OES_standard_derivatives : enable

out vec4 FragColor;

in vec2 texCoord;

float gridThickness = .08;

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
    FragColor = vec4(0.8, 0.8, 0.8, 1.0) * gridSmooth(texCoord);
}
)%==%";

namespace cudarrows {
    class GridShader : public BaseShader {
    public:
        gl::UniformMatrix4fv transform;

        GridShader() : BaseShader(gridVertex, gridFragment),
            transform(program.getUniform<gl::UniformMatrix4fv>("transform")) {}
    };
};