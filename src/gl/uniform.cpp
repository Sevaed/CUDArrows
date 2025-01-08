#include "gl/uniform.h"

void gl::Uniform1i::set(GLint v0) {
    glUniform1i(uniformLocation, v0);
}

void gl::Uniform1iv::set(GLsizei count, const GLint *value) {
    glUniform1iv(uniformLocation, count, value);
}

void gl::Uniform2i::set(GLint v0, GLint v1) {
    glUniform2i(uniformLocation, v0, v1);
}

void gl::Uniform2iv::set(GLsizei count, const GLint *value) {
    glUniform2iv(uniformLocation, count, value);
}

void gl::Uniform3i::set(GLint v0, GLint v1, GLint v2) {
    glUniform3i(uniformLocation, v0, v1, v2);
}

void gl::Uniform3iv::set(GLsizei count, const GLint *value) {
    glUniform3iv(uniformLocation, count, value);
}

void gl::Uniform4i::set(GLint v0, GLint v1, GLint v2, GLint v3) {
    glUniform4i(uniformLocation, v0, v1, v2, v3);
}

void gl::Uniform4iv::set(GLsizei count, const GLint *value) {
    glUniform4iv(uniformLocation, count, value);
}

void gl::Uniform1ui::set(GLuint v0) {
    glUniform1ui(uniformLocation, v0);
}

void gl::Uniform1uiv::set(GLsizei count, const GLuint *value) {
    glUniform1uiv(uniformLocation, count, value);
}

void gl::Uniform2ui::set(GLuint v0, GLuint v1) {
    glUniform2ui(uniformLocation, v0, v1);
}

void gl::Uniform2uiv::set(GLsizei count, const GLuint *value) {
    glUniform2uiv(uniformLocation, count, value);
}

void gl::Uniform3ui::set(GLuint v0, GLuint v1, GLuint v2) {
    glUniform3ui(uniformLocation, v0, v1, v2);
}

void gl::Uniform3uiv::set(GLsizei count, const GLuint *value) {
    glUniform3uiv(uniformLocation, count, value);
}

void gl::Uniform4ui::set(GLuint v0, GLuint v1, GLuint v2, GLuint v3) {
    glUniform4ui(uniformLocation, v0, v1, v2, v3);
}

void gl::Uniform4uiv::set(GLsizei count, const GLuint *value) {
    glUniform4uiv(uniformLocation, count, value);
}

void gl::Uniform1f::set(GLfloat v0) {
    glUniform1f(uniformLocation, v0);
}

void gl::Uniform1fv::set(GLsizei count, const GLfloat *value) {
    glUniform1fv(uniformLocation, count, value);
}

void gl::Uniform2f::set(GLfloat v0, GLfloat v1) {
    glUniform2f(uniformLocation, v0, v1);
}

void gl::Uniform2fv::set(GLsizei count, const GLfloat *value) {
    glUniform2fv(uniformLocation, count, value);
}

void gl::Uniform3f::set(GLfloat v0, GLfloat v1, GLfloat v2) {
    glUniform3f(uniformLocation, v0, v1, v2);
}

void gl::Uniform3fv::set(GLsizei count, const GLfloat *value) {
    glUniform3fv(uniformLocation, count, value);
}

void gl::Uniform4f::set(GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    glUniform4f(uniformLocation, v0, v1, v2, v3);
}

void gl::Uniform4fv::set(GLsizei count, const GLfloat *value) {
    glUniform4fv(uniformLocation, count, value);
}

void gl::Uniform1d::set(GLdouble v0) {
    glUniform1d(uniformLocation, v0);
}

void gl::Uniform1dv::set(GLsizei count, const GLdouble *value) {
    glUniform1dv(uniformLocation, count, value);
}

void gl::Uniform2d::set(GLdouble v0, GLdouble v1) {
    glUniform2d(uniformLocation, v0, v1);
}

void gl::Uniform2dv::set(GLsizei count, const GLdouble *value) {
    glUniform2dv(uniformLocation, count, value);
}

void gl::Uniform3d::set(GLdouble v0, GLdouble v1, GLdouble v2) {
    glUniform3d(uniformLocation, v0, v1, v2);
}

void gl::Uniform3dv::set(GLsizei count, const GLdouble *value) {
    glUniform3dv(uniformLocation, count, value);
}

void gl::Uniform4d::set(GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3) {
    glUniform4d(uniformLocation, v0, v1, v2, v3);
}

void gl::Uniform4dv::set(GLsizei count, const GLdouble *value) {
    glUniform4dv(uniformLocation, count, value);
}

void gl::UniformMatrix2fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix2fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix2x3fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix2x3fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix2x4fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix2x4fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix3x2fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix3x2fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix3fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix3fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix3x4fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix3x4fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix4x2fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix4x2fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix4x3fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix4x3fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix4fv::set(GLsizei count, bool transpose, const GLfloat *value) {
    glUniformMatrix4fv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix2dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix2dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix2x3dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix2x3dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix2x4dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix2x4dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix3x2dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix3x2dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix3dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix3dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix3x4dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix3x4dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix4x2dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix4x2dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix4x3dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix4x3dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}

void gl::UniformMatrix4dv::set(GLsizei count, bool transpose, const GLdouble *value) {
    glUniformMatrix4dv(uniformLocation, count, transpose ? GL_TRUE : GL_FALSE, value);
}