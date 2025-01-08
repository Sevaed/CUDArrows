#pragma once

#define MIN_SCALE 0.05
#define MAX_SCALE 2.0

namespace cudarrows {
    class Camera {
    private:
        float scale;
    
    public:
        float xOffset;
        float yOffset;

        Camera(float x, float y, float initialScale) : xOffset(x), yOffset(y), scale(initialScale) {}

        float getScale();

        void setScale(float newScale, float xOrigin, float yOrigin);
    };
}