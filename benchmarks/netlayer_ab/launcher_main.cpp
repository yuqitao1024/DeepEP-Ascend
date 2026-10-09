#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::vector<std::string> arguments(int argc, char** argv) {
    std::vector<std::string> result;
    for (int index = 1; index < argc; ++index)
        result.push_back(argv[index]);
    return result;
}

}  // namespace

extern int netlayer_ab_main(const std::vector<std::string>& arguments);

int main(int argc, char** argv) {
    return netlayer_ab_main(arguments(argc, argv));
}
