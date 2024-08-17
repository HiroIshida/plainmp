## modification note (Hirokazu Ishida)
1. this directory is a subdirecory of [urdfdom](https://github.com/ros/urdfdom). `CMakeLists.txt` is heavily modified. 
2. originally, only `urdf_parser` directory exist under the `include` directory. I added other include directories from [urdfdom_headers](https://github.com/ros/urdfdom_headers). 
3. external directory includes libraries required to build urdfdom [tinyxml](http://www.grinninglizard.com/tinyxml/). As for `tinyxml`, `CMakeLists.txt` is added in order to build everything via cmake. 
4. tinyxml is fetched from https://sourceforge.net/projects/tinyxml/ and modified referring https://stackoverflow.com/questions/17603822/can-not-pass-string-in-setattribute
5. applied formatter

## License
urdfdof -> BSD license
tinyxml -> zlib license
