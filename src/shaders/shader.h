#pragma once
#include <glad/glad.h>
#include "gl/uniform.h"
#include "gl/shader.h"

namespace cudarrows {
    class BaseShader {
    protected:
        gl::ShaderProgram program;

    public:
        gl::UniformMatrix4fv model;
        gl::UniformMatrix4fv view;
        gl::UniformMatrix4fv projection;

        BaseShader(const GLchar *vertexSource, const GLchar *fragmentSource) :
            program(gl::ShaderProgram(
                gl::Shader(gl::ShaderType::Vertex, vertexSource),
                gl::Shader(gl::ShaderType::Fragment, fragmentSource)
            )),
            model(program.getUniform<gl::UniformMatrix4fv>("model")),
            view(program.getUniform<gl::UniformMatrix4fv>("view")),
            projection(program.getUniform<gl::UniformMatrix4fv>("projection")) {}

        void use();
    };
};