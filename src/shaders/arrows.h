#pragma once
#include "gl/uniform.h"
#include "shaders/shader.h"

const char *arrowsVertex = R"%==%(
#version 330 core
layout (location = 0) in vec4 position;

out vec2 uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    uv = position.xy;
    gl_Position = projection * view * model * position;
}
)%==%";

const char *arrowsFragment = R"%==%(
#version 330 core

out vec4 FragColor;

in vec2 uv;

uniform sampler2D arrowAtlas;
uniform sampler2D data;

void main() {
    FragColor = vec4(uv, 0.0, 1.0);
    /*vec2 texSize = vec2(textureSize(data, 0));
    vec4 arrowData = texture2D(data, floor(texCoord) / texSize);
    if (int(arrowData.x * 256.0) == 1) {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }*/
}
)%==%";

namespace cudarrows {
    class ArrowsShader : public BaseShader {
    public:
        gl::Uniform1i arrowAtlas;
        gl::Uniform1i data;
        gl::UniformMatrix4fv model;
        gl::UniformMatrix4fv view;
        gl::UniformMatrix4fv projection;

        ArrowsShader() : BaseShader(arrowsVertex, arrowsFragment),
            arrowAtlas(program.getUniform<gl::Uniform1i>("arrowAtlas")),
            data(program.getUniform<gl::Uniform1i>("data")),
            model(program.getUniform<gl::UniformMatrix4fv>("model")),
            view(program.getUniform<gl::UniformMatrix4fv>("view")),
            projection(program.getUniform<gl::UniformMatrix4fv>("projection")) {}
    };
};