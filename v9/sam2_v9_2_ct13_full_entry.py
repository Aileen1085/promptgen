"""Combine full SAM2 optimization with the fixed CT13 evaluation protocol."""

import finetune_multisource_sam2_v9_2_full as full  # Installs encoder/decoder hooks.
from sam2_v9_2_ct13_entry import main as ct13_main


def main():
    return ct13_main()


if __name__ == "__main__":
    main()
