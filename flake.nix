{
  description = "DeepVQE development utilities";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.squashfsTools # mksquashfs, unsquashfs
          pkgs.squashfuse    # squashfuse (FUSE mount)
          pkgs.cmake         # GGML C++ build
          pkgs.gcc           # C/C++ compiler
          pkgs.pkg-config
          pkgs.libsndfile    # FLAC/WAV decoding for tests
          pkgs.linuxPackages.perf  # CPU profiling
        ];
      };
    };
}
