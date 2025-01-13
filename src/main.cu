#include <exception>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "util/cuda_assert.cuh"
#include "shaders/arrows.h"
#include "shaders/grid.h"
#include "camera.h"
#include "map.cuh"

#define CELL_SIZE 64.0f

namespace fs = std::filesystem;

static void cudarrows_terminate() {
    try {
        if (std::current_exception())
            std::rethrow_exception(std::current_exception());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
    }
    abort();
}

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

GLsizei roundToPowerOf2(GLsizei n) {
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return ++n < 16 ? 16 : n;
}

int main(int argc, char *argv[]) {
    std::set_terminate(cudarrows_terminate);

    if (argc < 2) {
        printf("Usage: %s <map-code>\n", argv[0]);
        return 1;
    }

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "CUDArrows", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return 1;
    }

#ifndef NDEBUG
    glEnable(GL_DEBUG_OUTPUT);
#endif

    GLfloat vertices[] = {
        1.f, 1.f, 0.f, 1.f,
        1.f, 0.f, 0.f, 1.f,
        0.f, 0.f, 0.f, 1.f,
        0.f, 1.f, 0.f, 1.f,
    };

    GLuint indices[] = {
        0, 1, 3,
        1, 2, 3,
    };

    GLuint VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW); 

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    fs::path resourcesPath(argv[0]);
    resourcesPath = resourcesPath.parent_path();
    resourcesPath.append("res");

    fs::path atlasPath(resourcesPath);
    atlasPath.append("atlas.png");

    fs::path fontPath(resourcesPath);
    fontPath.append("Nunito.ttf");

    int atlasWidth, atlasHeight;
    GLubyte *atlasData = stbi_load(atlasPath.string().c_str(), &atlasWidth, &atlasHeight, NULL, 0);
    if (!atlasData) {
        fprintf(stderr, "Failed load texture atlas\n");
        return 1;
    }

    GLuint atlasTexture;

    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &atlasTexture);
    glBindTexture(GL_TEXTURE_2D, atlasTexture);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlasWidth, atlasHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, atlasData);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(atlasData);

    cudarrows::Map map(argv[1]);

    cudarrows::ArrowsShader arrows;
    cudarrows::GridShader grid;

    arrows.arrowAtlas.set(1);
    arrows.data.set(0);

    cudarrows::Camera camera(0.f, 0.f, 1.f);

    map.reset(time(NULL));

    float lastCameraX = camera.xOffset, lastCameraY = camera.yOffset;

    glm::mat4 projection = glm::ortho(0.f, 1.f, 1.f, 0.f);

    arrows.use();
    arrows.projection.set(1, false, glm::value_ptr(projection));
    arrows.arrowAtlas.set(1);
    arrows.data.set(0);

    grid.use();
    grid.projection.set(1, false, glm::value_ptr(projection));

    cudaGraphicsResource_t cudaTexture = nullptr;

    GLuint dataTexture;

    uint8_t fill[4] = { 0, 0, 0, 255 };

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    GLsizei texWidth, texHeight,
            lastSpanX, lastSpanY;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    io.Fonts->AddFontFromFileTTF(fontPath.string().c_str(), 24.0f);

    bool controlsWindowVisible = true,
         debugWindowVisible = false;

    int targetTPS = 3;

    bool playing = true,
         doStepFlag = false;

    double nextUpdate = glfwGetTime();

    bool buttonHovered = false;

    cudaEvent_t updateStart, updateEnd;
    cuda_assert(cudaEventCreate(&updateStart));
    cuda_assert(cudaEventCreate(&updateEnd));

    unsigned long nticks = 0;

    float tps = 0.f;

    cuda_assert(cudaEventRecord(updateStart));

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            nextUpdate = glfwGetTime() + 1.0 / targetTPS;
            continue;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (!io.WantCaptureMouse) {
            bool mouseMoved = io.MousePos.x != io.MousePosPrev.x || io.MousePos.y != io.MousePosPrev.y;

            if (io.MouseDown[2] && mouseMoved) {
                camera.xOffset = lastCameraX + io.MousePos.x - io.MousePosPrev.x;
                camera.yOffset = lastCameraY + io.MousePos.y - io.MousePosPrev.y;
            }

            int32_t arrowX = int32_t(floor((-camera.xOffset + io.MousePos.x) / CELL_SIZE / camera.getScale()));
            int32_t arrowY = int32_t(floor((-camera.yOffset + io.MousePos.y) / CELL_SIZE / camera.getScale()));

            if (!io.MouseDown[2] && mouseMoved) {
                cudarrows::ArrowInfo arrow = map.getArrow(arrowX, arrowY);

                buttonHovered = arrow.type == cudarrows::ArrowType::Button || arrow.type == cudarrows::ArrowType::DirectionalButton;
            }

            if (io.MouseReleased[0] && buttonHovered) {
                cudarrows::ArrowInput input;
                input.buttonPressed = true;

                map.sendInput(arrowX, arrowY, input);
            }

            ImGui::SetMouseCursor(buttonHovered ? ImGuiMouseCursor_Hand : ImGuiMouseCursor_Arrow);
        }

        if (io.MouseWheel > 0.0)
            camera.setScale(camera.getScale() * io.MouseWheel * 1.2f, io.MousePos.x, io.MousePos.y);
        else if (io.MouseWheel < 0.0)
            camera.setScale(camera.getScale() / -io.MouseWheel / 1.2f, io.MousePos.x, io.MousePos.y);

        lastCameraX = camera.xOffset;
        lastCameraY = camera.yOffset;

        if (targetTPS < 1)
            targetTPS = 1;

        if (playing) {
            while (glfwGetTime() >= nextUpdate) {
                map.update();
                ++nticks;
                nextUpdate += 1.0 / targetTPS;
            }
        } else {
            if (doStepFlag) {
                map.update();
                doStepFlag = false;
            }
            nextUpdate = glfwGetTime() + 1.0 / targetTPS;
            tps = 0.f;
        }

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        int32_t minX = int32_t(-camera.xOffset / camera.getScale() / CELL_SIZE) - 1,
                minY = int32_t(-camera.yOffset / camera.getScale() / CELL_SIZE) - 1,
                maxX = int32_t((-camera.xOffset + io.DisplaySize.x) / camera.getScale() / CELL_SIZE),
                maxY = int32_t((-camera.yOffset + io.DisplaySize.y) / camera.getScale() / CELL_SIZE);
        
        GLsizei spanX = GLsizei(io.DisplaySize.x / camera.getScale() / CELL_SIZE) + 2,
                spanY = GLsizei(io.DisplaySize.y / camera.getScale() / CELL_SIZE) + 2;

        if (cudaTexture == nullptr || lastSpanX != spanX || lastSpanY != spanY) {
            lastSpanX = spanX;
            lastSpanY = spanY;

            GLsizei newTexWidth = roundToPowerOf2(spanX),
                    newTexHeight = roundToPowerOf2(spanY);
            if (cudaTexture == nullptr || newTexWidth != texWidth || newTexHeight != texHeight) {
                texWidth = newTexWidth;
                texHeight = newTexHeight;

                glActiveTexture(GL_TEXTURE0);

                if (cudaTexture != nullptr) {
                    cuda_assert(cudaGraphicsUnregisterResource(cudaTexture));

                    glDeleteTextures(1, &dataTexture);
                }

                glGenTextures(1, &dataTexture);
                glBindTexture(GL_TEXTURE_2D, dataTexture);
                
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                glGenerateMipmap(GL_TEXTURE_2D);

                cuda_assert(cudaGraphicsGLRegisterImage(&cudaTexture, dataTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
            }
        }

        glClearTexImage(dataTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, &fill);

        cudaArray_t cuda_array;

        cuda_assert(cudaGraphicsMapResources(1, &cudaTexture));
        cuda_assert(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cudaTexture, 0, 0));
        
        resDesc.res.array.array = cuda_array;
        cudaSurfaceObject_t surface;
        cuda_assert(cudaCreateSurfaceObject(&surface, &resDesc));

        map.render(surface, minX, minY, maxX, maxY);

        cuda_assert(cudaDestroySurfaceObject(surface));

        cuda_assert(cudaGraphicsUnmapResources(1, &cudaTexture));
        
        if (nticks > 0) {
            cuda_assert(cudaEventRecord(updateEnd));

            cuda_assert(cudaEventSynchronize(updateEnd));

            float elapsedTime;
            cuda_assert(cudaEventElapsedTime(&elapsedTime, updateStart, updateEnd));
            tps = nticks * 1000.f / elapsedTime;
            nticks = 0;

            cuda_assert(cudaEventRecord(updateStart));
        }

        cuda_assert(cudaDeviceSynchronize());

        glm::mat4 view(1.f);
        view = glm::translate(view, glm::vec3(
            camera.xOffset / io.DisplaySize.x,
            camera.yOffset / io.DisplaySize.y,
            0.f
        ));
        view = glm::scale(view, glm::vec3(camera.getScale(), camera.getScale(), 1.f));

        glm::mat4 model(1.f);
        model = glm::translate(model, glm::vec3(
            CELL_SIZE * minX / io.DisplaySize.x,
            CELL_SIZE * minY / io.DisplaySize.y,
            0.f
        ));
        model = glm::scale(model, glm::vec3(CELL_SIZE * texWidth / io.DisplaySize.x, CELL_SIZE * texHeight / io.DisplaySize.y, 1.f));

        arrows.use();
        arrows.view.set(1, false, glm::value_ptr(view));
        arrows.model.set(1, false, glm::value_ptr(model));
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        grid.use();
        grid.view.set(1, false, glm::value_ptr(view));
        grid.model.set(1, false, glm::value_ptr(model));
        grid.tileCount.set(texWidth, texHeight);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("App")) {
                if (ImGui::MenuItem("Exit", "Alt+F4")) {
                    break;
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Controls", NULL, &controlsWindowVisible);

                ImGui::MenuItem("Debug", NULL, &debugWindowVisible);

                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();
        }

        if (controlsWindowVisible) {
            ImGui::Begin("Controls", &controlsWindowVisible);

            ImGui::Checkbox("Playing", &playing);

            if (!playing) {
                ImGui::SameLine();

                if (ImGui::Button("Step")) {
                    doStepFlag = true;
                }
            }

            ImGui::InputInt("TPS", &targetTPS);

            if (ImGui::Button("Reset map")) {
                map.reset(time(NULL));
            }

            ImGui::End();
        }

        if (debugWindowVisible) {
            ImGui::Begin("Debug", &debugWindowVisible);

            ImGui::Text("Game is running at %.1f TPS (%.1f FPS)", tps, io.Framerate);

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    cuda_assert(cudaGraphicsUnregisterResource(cudaTexture));

    cuda_assert(cudaEventDestroy(updateStart));
    cuda_assert(cudaEventDestroy(updateEnd));

    glDeleteTextures(1, &atlasTexture);
    glDeleteTextures(1, &dataTexture);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}