"""Check HF token and (gated) access for every model repo in configs/models.yaml.

Exits 0 when every repo is accessible, 1 otherwise (FLUX gated denial is a spec stop condition).
"""

import argparse
import sys

from huggingface_hub import auth_check, whoami
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from fmsr.config import load_models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="configs/models.yaml")
    args = p.parse_args()

    user = whoami()["name"]
    print(f"HF user: {user}")
    ok = True
    for name, variant in load_models(args.models).variants.items():
        try:
            auth_check(variant.hf_repo)
            print(f"OK      {name}: {variant.hf_repo}")
        except GatedRepoError:
            ok = False
            print(f"GATED   {name}: {variant.hf_repo} (accept the license on huggingface.co with user {user})")
        except RepositoryNotFoundError:
            ok = False
            print(f"MISSING {name}: {variant.hf_repo}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
