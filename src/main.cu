#include <exception>
#include <cstdlib>
#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shaders/grid.h"
#include "camera.h"
#include "map.h"
#include "chunkupdates.h"
#include "render.h"

#define CELL_SIZE 64.0

float scroll = 0.0;

static void cudarrows_terminate() {
    try {
        if (std::current_exception())
            std::rethrow_exception(std::current_exception());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
    }
    std::abort();
}

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    scroll += static_cast<float>(yoffset);
}

static void cuda_assert(cudaError_t error, bool fatal = true) {
    if (error != cudaError::cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s\n", error, cudaGetErrorString(error));
        if (fatal) exit(1);
    }
}

static void cuda_report(cudaError_t error) {
    cuda_assert(error, false);
}

int main(void) {
    std::set_terminate(cudarrows_terminate);

    cudaStream_t stream;
    cuda_assert(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    cuda_assert(cudaEventCreate(&start));
    cuda_assert(cudaEventCreate(&stop));

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "CUDArrows", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return 1;
    }

    float vertices[] = {
        0.f, 0.f,
        1.f, 0.f,
        1.f, 1.f,
        0.f, 0.f,
        1.f, 1.f,
        0.f, 1.f,
    };

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    cudarrows::Map map;

    cudarrows::GridShader grid;

    cudarrows::Camera camera(0.f, 0.f, 1.f);

    map.load("AAABAAAAAAAAAQAAAA==");

    double lastMouseX, lastMouseY;
    float lastCameraX, lastCameraY;

    uint8_t step = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
            continue;

        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        bool wheelDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

        if (wheelDown) {
            camera.xOffset = static_cast<float>(lastCameraX + mouseX - lastMouseX);
            camera.yOffset = static_cast<float>(lastCameraY + mouseY - lastMouseY);
        }

        if (scroll > 0.0)
            camera.setScale(camera.getScale() * scroll * 1.2f, static_cast<float>(mouseX), static_cast<float>(mouseY));
        else if (scroll < 0.0)
            camera.setScale(camera.getScale() / -scroll / 1.2f, static_cast<float>(mouseX), static_cast<float>(mouseY));
        scroll = 0.0f;

        lastCameraX = camera.xOffset;
        lastCameraY = camera.yOffset;
        lastMouseX = mouseX;
        lastMouseY = mouseY;

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        uint32_t minX = uint32_t(-camera.xOffset / camera.getScale() / CELL_SIZE);
        uint32_t minY = uint32_t(-camera.yOffset / camera.getScale() / CELL_SIZE);
        uint32_t maxX = uint32_t((-camera.xOffset + width) / camera.getScale() / CELL_SIZE);
        uint32_t maxY = uint32_t((-camera.yOffset + height) / camera.getScale() / CELL_SIZE);
        uint32_t spanX = maxX - minX;
        uint32_t spanY = maxY - minY;

        printf("render %dx%d from (%d; %d) to (%d; %d)\n", spanX, spanY, minX, minY, maxX, maxY);

        render<<<map.countChunks(), dim3(CHUNK_SIZE, CHUNK_SIZE)>>>(0, map.getChunks(), step, minX, minY, maxX, maxY);

        glm::mat4 cameraTransform(1.f);
        cameraTransform = glm::translate(cameraTransform, glm::vec3(
            -camera.xOffset / camera.getScale() / CELL_SIZE,
            -camera.yOffset / camera.getScale() / CELL_SIZE,
            0.f
        ));
        cameraTransform = glm::scale(cameraTransform, glm::vec3(
            width / camera.getScale() / CELL_SIZE,
            height / camera.getScale() / CELL_SIZE,
            1.f
        ));

        grid.use();
        grid.transform.set(1, false, glm::value_ptr(cameraTransform));
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}