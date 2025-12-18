{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.python312Packages.pip
            pkgs.python312Packages.setuptools
            pkgs.python312Packages.virtualenv
            pkgs.python312Packages.pyaudio
            pkgs.ffmpeg_6
            pkgs.portaudio
            pkgs.ffmpeg_6.dev
          ];
          
          shellHook = ''
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
              with pkgs;
              lib.makeLibraryPath [ 
                stdenv.cc.cc.lib
                portaudio
                udev
                SDL2.dev
                zlib
              ]
            }"
            # Activate python venv
            source env/bin/activate
          '';
        };
      });
}
