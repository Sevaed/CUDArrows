#pragma once
#include <string>
#include <glad/glad.h>
#include "gl/glerror.h"

namespace gl {
    enum ShaderType : GLenum {
        Vertex = GL_VERTEX_SHADER,
        Fragment = GL_FRAGMENT_SHADER
    };

    class GLShaderCompileError : public GLError {
    private:
        static std::string getInfoLog(GLuint glShader);

    public:
        GLShaderCompileError(GLuint glShader) : GLError(getInfoLog(glShader)) {}
    };

    class GLShaderProgramLinkError : public GLError {
    private:
        static std::string getInfoLog(GLuint glShaderProgram);

    public:
        GLShaderProgramLinkError(GLuint glShaderProgram) : GLError(getInfoLog(glShaderProgram)) {}
    };

    class Shader {
    private:
        GLuint shader;

    public:
        Shader(ShaderType type, const GLchar *source);

        ~Shader();

        GLuint glShader() const { return shader; }
    };

    class ShaderProgram {
    private:
        GLuint shaderProgram;

    public:
        ShaderProgram(Shader vertexShader, Shader fragmentShader);

        ~ShaderProgram();

        GLuint glShaderProgram() const { return shaderProgram; }

        template <class T> T getUniform(const GLchar *name) {
            return T(glGetUniformLocation(shaderProgram, name));
        }

        void use();
    };
};