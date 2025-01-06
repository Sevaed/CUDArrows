#pragma once
#include <glad/glad.h>
#include "gl/shader.h"

namespace cudarrows {
    class BaseShader {
    protected:
        gl::ShaderProgram program;

    public:
        BaseShader(const GLchar *vertexSource, const GLchar *fragmentSource) :
            program(gl::ShaderProgram(
                gl::Shader(gl::ShaderType::Vertex, vertexSource),
                gl::Shader(gl::ShaderType::Fragment, fragmentSource)
            )) {}

        void use();
    };
};