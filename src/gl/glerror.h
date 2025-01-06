#pragma once
#include <string>
#include <exception>
#include <glad/glad.h>

namespace gl {
    class GLError : public std::exception {
    protected:
        std::string infoLog;

    public:
        GLError(const std::string& info) : infoLog(info) {}

        const char* what() const {
            return infoLog.c_str();
        }
    };
};