#!/usr/bin/env bash
# The framework ships WITHOUT a brain. Provide your own GGUF:
#   * download any GGUF (HuggingFace, etc.) into ~/ or ~/Downloads, then in the TUI
#     run  /gguf  to pick it, or  kilo brain deploy /path/to/model.gguf
#   * or just use a cloud model:  /cloud  (OpenRouter, OpenAI, Anthropic, ... )
echo 'kilobyte-framework ships without a brain. Use /gguf to pick a downloaded GGUF,'
echo 'or /cloud to use a hosted model. See the README.'
