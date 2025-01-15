#include "logicarrows/client.h"

logicarrows::MapInfo logicarrows::Client::getMap(const std::string &id) {
    json body = {
        { "id", id }
    };
    cpr::Response r = cpr::Post(cpr::Url{API_ENDPOINT "/map"},
                               cpr::Cookies{{"accessToken", token}},
                               cpr::Header{{"User-Agent", userAgent}, {"Content-Type", "application/json"}},
                               cpr::Body{body.dump()});
    json resp = json::parse(r.text);
    return logicarrows::MapInfo {
        resp["id"],
        resp["name"],
        resp["data"]
    };
}