<div align="center">
  <img src="media/cudarrows-logo.png"/>
</div>

<div align="center">
    <i>CUDA port of <a href="https://logic-arrows.io">Logic Arrows</a></i>
    <hr>
</div>

## About

CUDArrows is a port of online game [Logic Arrows](https://logic-arrows.io) created by [Onigiri](https://www.youtube.com/@OnigiriScience). It uses [NVIDIA® CUDA®](https://developer.nvidia.com/cuda-toolkit) to enable hardware acceleration of the cellular automaton. CUDA allows to perform computation of every arrow on the map in parallel. This maximizes the performance and theoretically provides the fastest speed that can be achieved.

## Installing

You can either download [pre-built binaries](https://github.com/sagdrip/CUDArrows/releases) for your system or build from source.

### Dependencies

Building CUDArrows requires the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and the [vcpkg](https://github.com/microsoft/vcpkg) package manager. On Ubuntu you can install CUDA via your package manager or the official installer. After installing CUDA clone and bootstrap vcpkg:

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
export VCPKG_ROOT="$(pwd)"
```

### Building from source

Clone this GitHub repository and use CMake to build the project. Provide the vcpkg toolchain file during configuration:
```bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config=Release
cmake --install . --prefix /path/to/cudarrows --strip
```

## Using

CUDArrows is a GUI application, however it must be launched from the command line. You must specify map save code or URL as a CLI argument:
```
cudarrows AAABAAAAAAADBh4AABAAIABAAFAAUQACABIAIgBCAFIAVAAFARUBJQFVAAcAFwBnACgBCgAaAEoAWgArAWsBHQBNAF0AHgBeAAcOYABiAwQBJABEAUUCZQNHAUgCaAINAQ4CLgNOAm4DAgkRAWEBFAFkAScBWAEqAWoBLQFtAQgAVwE=
```

**CUDArrows is not an editor, it is only a player**. This means that you cannot modify the map in any way using CUDArrows. To load a different map, simply restart the app and specify the new save code.

### Logging in

Logic Arrows API doesn't permit unauthorized users to download map data. Because of that, you must first log in to load maps by URL.

#### Using the CLI tool

CUDArrows provide a CLI tool for your convinience:
```
py cudarrows-login.py
```
This tool launches Chrome (or Firefox if you specify it in the `--browser` argument), waits until you log into Logic Arrows website using your Google account and then creates a `session.txt` file for you. Note that you must launch CUDArrows from the same directory you have launched `cudarrows-login` from, because CUDArrows look for `session.txt` file in the current working directory.

#### Manually

You can also create `session.txt` manually. To do this, go to [Logic Arrows website](https://logic-arrows.io), open developer tools and copy your user agent and `accessToken` cookie. Then, create a `session.txt` file containing the UA in the first line and access token in the second line.

### Configuring the runtime

When you launch the app, you will see the game grid and the control panel. You can use it to pause/resume the game, perform single steps, configure the tick rate and resetting the map. It's similar to Logic Arrows controls, except it uses GUI window instead of keybinds.

### Debugging

You can also open the `Debug` window to see debug information such as FPS and TPS, targetted arrow, etc. **Warning:** this panel might get deprecated, because it requires CUDArrows to perform additional actions on the GPU, which may or may not slow down the entire app. It will always be available in debug builds, which you can compile from source.

## Contributing

Feel free to open issues and pull requests or fork this repository! Logic Arrows have a very creative community, and that's one of the reasons I love this game. You can also contact me in [Telegram](https://t.me/sagdrip) directly or join the [Onigiri's official TG group](https://t.me/onigiriscichat) and chat there if you have come up with an optimization idea or if you have a suggestion.