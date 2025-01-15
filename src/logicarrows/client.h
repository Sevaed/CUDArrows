#pragma once
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define API_ENDPOINT "http://logic-arrows.io/api"

namespace logicarrows {
    struct MapInfo {
        std::string id;
        std::string name;
        std::string data;
    };

    class Client {
    private:
        const std::string &userAgent;
        const std::string &token;
        bool authorized = false;

    public:
        Client(const std::string &userAgent, const std::string &token) : userAgent(userAgent), token(token) {}

        MapInfo getMap(const std::string &id);
    };
}