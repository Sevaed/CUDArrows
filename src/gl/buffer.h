#pragma once
#include <glad/glad.h>

namespace gl {
    enum BufferTarget : GLenum {
        Array = GL_ARRAY_BUFFER,
        ElementArray = GL_ELEMENT_ARRAY_BUFFER
    };

    enum BufferUsage : GLenum {
        ReadOnly = GL_READ_ONLY,
        WriteOnly = GL_WRITE_ONLY,
        BufferAccess = GL_BUFFER_ACCESS,
        BufferMapped = GL_BUFFER_MAPPED,
        BufferMapPointer = GL_BUFFER_MAP_POINTER,
        StreamDraw = GL_STREAM_DRAW,
        StreamRead = GL_STREAM_READ,
        StreamCopy = GL_STREAM_COPY,
        StaticDraw = GL_STATIC_DRAW,
        StaticRead = GL_STATIC_READ,
        StaticCopy = GL_STATIC_COPY,
        DynamicDraw = GL_DYNAMIC_DRAW,
        DynamicRead = GL_DYNAMIC_READ,
        DynamicCopy = GL_DYNAMIC_COPY
    };

    class Buffer {
    protected:
        gl::BufferTarget target;
        GLuint bufferObject;

    public:
        Buffer(gl::BufferTarget target);

        ~Buffer();

        void bind();

        void data(GLsizeiptr size, const void *data, gl::BufferUsage usage);

        GLuint glBufferObject() const { return bufferObject; }
    };
}