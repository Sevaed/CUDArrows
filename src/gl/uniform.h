#pragma once
#include <glad/glad.h>

namespace gl {
    class Uniform {
    protected:
        GLint uniformLocation;

    public:
        Uniform(GLint location) : uniformLocation(location) {}

        GLuint glUniformLocation() const { return uniformLocation; }
    };

    class Uniform1i : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLint v0);
    };

    class Uniform1iv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLint *value);
    };

    class Uniform2i : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLint v0, GLint v1);
    };

    class Uniform2iv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLint *value);
    };

    class Uniform3i : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLint v0, GLint v1, GLint v2);
    };

    class Uniform3iv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLint *value);
    };

    class Uniform4i : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLint v0, GLint v1, GLint v2, GLint v3);
    };

    class Uniform4iv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLint *value);
    };

    class Uniform1ui : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLuint v0);
    };

    class Uniform1uiv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLuint *value);
    };

    class Uniform2ui : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLuint v0, GLuint v1);
    };

    class Uniform2uiv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLuint *value);
    };

    class Uniform3ui : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLuint v0, GLuint v1, GLuint v2);
    };

    class Uniform3uiv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLuint *value);
    };

    class Uniform4ui : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLuint v0, GLuint v1, GLuint v2, GLuint v3);
    };

    class Uniform4uiv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLuint *value);
    };

    class Uniform1f : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLfloat v0);
    };

    class Uniform1fv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLfloat *value);
    };

    class Uniform2f : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLfloat v0, GLfloat v1);
    };

    class Uniform2fv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLfloat *value);
    };

    class Uniform3f : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLfloat v0, GLfloat v1, GLfloat v2);
    };

    class Uniform3fv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLfloat *value);
    };

    class Uniform4f : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
    };

    class Uniform4fv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLfloat *value);
    };

    class Uniform1d : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLdouble v0);
    };

    class Uniform1dv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLdouble *value);
    };

    class Uniform2d : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLdouble v0, GLdouble v1);
    };

    class Uniform2dv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLdouble *value);
    };

    class Uniform3d : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLdouble v0, GLdouble v1, GLdouble v2);
    };

    class Uniform3dv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLdouble *value);
    };

    class Uniform4d : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3);
    };

    class Uniform4dv : public Uniform {
    public:
        using Uniform::Uniform;

        void set(GLsizei count, const GLdouble *value);
    };
};