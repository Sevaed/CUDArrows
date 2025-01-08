#include "gl/shader.h"

std::string gl::GLShaderCompileError::getInfoLog(GLuint glShader) {
    GLint infoLogLength;
    glGetShaderiv(glShader, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength == 0) return "";
    std::string infoLog(infoLogLength - 1, 0);
    glGetShaderInfoLog(glShader, infoLogLength - 1, NULL, &infoLog[0]);
    return infoLog;
}

std::string gl::GLShaderProgramLinkError::getInfoLog(GLuint glShaderProgram) {
    GLint infoLogLength;
    glGetProgramiv(glShaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength == 0) return "";
    std::string infoLog(infoLogLength - 1, 0);
    glGetProgramInfoLog(glShaderProgram, infoLogLength - 1, NULL, &infoLog[0]);
    return infoLog;
}

gl::Shader::Shader(ShaderType type, const GLchar *source) {
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
        throw GLShaderCompileError(shader);
}

gl::Shader::~Shader() {
    glDeleteShader(shader);
}

gl::ShaderProgram::ShaderProgram(Shader vertexShader, Shader fragmentShader) {
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader.glShader());
    glAttachShader(shaderProgram, fragmentShader.glShader());
    glLinkProgram(shaderProgram);
    
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
        throw GLShaderProgramLinkError(shaderProgram);
}

gl::ShaderProgram::~ShaderProgram() {
    glDeleteProgram(shaderProgram);
}

void gl::ShaderProgram::use() {
    glUseProgram(shaderProgram);
}