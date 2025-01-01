.PHONY: all clean
PROJECTDIR=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
SRCDIR=src
IMGUIDIR=imgui
NVCC=nvcc
CUFLAGS=-c -g -I"$(PROJECTDIR)/$(IMGUIDIR)" -I"$(PROJECTDIR)/$(IMGUIDIR)/backends"
LDFLAGS=
BUILDDIR=build
TARGET=$(BUILDDIR)/cudarrows.exe
SOURCES=$(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/*.cu)
SOURCES+=$(wildcard $(IMGUIDIR)/*.cpp) $(wildcard $(IMGUIDIR)/backends/imgui_impl_glfw.cpp) $(wildcard $(IMGUIDIR)/backends/imgui_impl_opengl3.cpp)
OBJECTS=$(subst $(IMGUIDIR)/,$(BUILDDIR)/,$(subst $(IMGUIDIR)/backends/,$(IMGUIDIR)/,$(subst $(SRCDIR)/,$(BUILDDIR)/,$(patsubst %.cu,%.o,$(patsubst %.cpp,%.o,$(SOURCES))))))

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Linux)
	ECHO_MESSAGE = "Linux"
	LDFLAGS += -lGL `pkg-config --static --libs glfw3`

	CUFLAGS += `pkg-config --cflags glfw3`
endif

ifeq ($(UNAME_S), Darwin)
	ECHO_MESSAGE = "Mac OS X"
	LDFLAGS += -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
	LDFLAGS += -L/usr/local/lib -L/opt/local/lib -L/opt/homebrew/lib
	LDFLAGS += -lglfw

	CUFLAGS += -I/usr/local/include -I/opt/local/include -I/opt/homebrew/include
endif

ifeq ($(OS), Windows_NT)
	ECHO_MESSAGE = "Windows"
	LDFLAGS += -L"$(PROJECTDIR)/glfw-win/lib-vc2022" -lglfw3_mt -luser32 -lgdi32 -lopengl32 -limm32

	CUFLAGS += -I"$(PROJECTDIR)/glfw-win/include"
endif

all: $(TARGET)
	@echo Build complete for $(ECHO_MESSAGE)

$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(NVCC) $(LDFLAGS) $+ -o $@

$(BUILDDIR)/%.o: $(IMGUIDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CUFLAGS) $< -o $@

$(BUILDDIR)/%.o: $(IMGUIDIR)/backends/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CUFLAGS) $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CUFLAGS) $< -o $@

clean:
	rm -rf $(BUILDDIR)/*