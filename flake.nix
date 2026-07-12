{
  description = "Development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-utils.url = "github:numtide/flake-utils";

    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      treefmt-nix,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        treefmtEval = treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true; # Nix
          programs.ruff-format.enable = true; # Python (model-runtime)
          programs.mix-format.enable = true; # Elixir (control-loop)
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            lefthook

            # Python (model-runtime — the mlx-vlm fork, fine-tuning)
            python3
            uv

            # Elixir (control-loop — ControlLoop, ActionQueue, emily)
            beamPackages.elixir
          ];
        };

        # `nix fmt` runs treefmt across the repo.
        formatter = treefmtEval.config.build.wrapper;

        # `nix flake check` verifies everything is formatted.
        checks.formatting = treefmtEval.config.build.check ./.;
      }
    );
}
