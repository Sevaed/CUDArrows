#include "camera.h"

double cudarrows::Camera::getScale() {
    return scale;
}

void cudarrows::Camera::setScale(double newScale, double xOrigin, double yOrigin) {
    if (newScale < MIN_SCALE)
        newScale = MIN_SCALE;
    if (newScale > MAX_SCALE)
        newScale = MAX_SCALE;
    xOffset = xOffset / scale * newScale + xOrigin - xOrigin / scale * newScale;
    yOffset = yOffset / scale * newScale + yOrigin - yOrigin / scale * newScale;
    scale = newScale;
}