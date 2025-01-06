#include "gl/buffer.h"

gl::Buffer::Buffer(gl::BufferTarget target) {
    this->target = target;
    glGenBuffers(1, &bufferObject);
}

gl::Buffer::~Buffer() {
    glDeleteBuffers(1, &bufferObject);
}

void gl::Buffer::bind() {
    glBindBuffer(target, bufferObject);
}

void gl::Buffer::data(GLsizeiptr size, const void *data, gl::BufferUsage usage) {
    glBufferData(target, size, data, usage);
}