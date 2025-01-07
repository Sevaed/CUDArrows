#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shaders/background.h"
#include "camera.h"
#include "map.h"
#include "chunkupdates.h"

#define CELL_SIZE 64.0

double scroll = 0.0;

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    scroll += yoffset;
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

    cudarrows::BackgroundShader background;

    cudarrows::Camera camera(0.0, 0.0, 1.0);

    map.load("AAABAAAAAAAAAQAAAA==");

    printf("arrow type: %d\n", map.getArrow(0, 0).type);

    bool lastWheelDown = false;

    double mouseStartX, mouseStartY, cameraStartX, cameraStartY;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
            continue;

        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        bool wheelDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        if (wheelDown && !lastWheelDown) {
            mouseStartX = mouseX;
            mouseStartY = mouseY;
            cameraStartX = camera.xOffset;
            cameraStartY = camera.yOffset;
        }
        lastWheelDown = wheelDown;

        if (wheelDown) {
            camera.xOffset = cameraStartX + mouseX - mouseStartX;
            camera.yOffset = cameraStartY + mouseY - mouseStartY;
        }

        if (scroll > 0.0)
            camera.setScale(camera.getScale() * scroll * 1.2, mouseX, mouseY);
        else if (scroll < 0.0)
            camera.setScale(camera.getScale() / -scroll / 1.2, mouseX, mouseY);
        scroll = 0.0;

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        background.use();
        background.transform.set(
            camera.xOffset / camera.getScale() / CELL_SIZE,
            camera.yOffset / camera.getScale() / CELL_SIZE,
            (double)width / camera.getScale() / CELL_SIZE,
            (double)height / camera.getScale() / CELL_SIZE
        );
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