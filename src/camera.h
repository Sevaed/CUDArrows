#pragma once

#define MIN_SCALE 0.05
#define MAX_SCALE 2.0

namespace cudarrows {
    class Camera {
    private:
        double scale;
    
    public:
        double xOffset;
        double yOffset;

        Camera(double x, double y, double initialScale) : xOffset(x), yOffset(y), scale(initialScale) {}

        double getScale();

        void setScale(double newScale, double xOrigin, double yOrigin);
    };
}