/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { theme } from '../semantic-colors.js';
import { RadioButtonSelect } from '../components/shared/RadioButtonSelect.js';
import {
  fetchOllamaModels,
  type OllamaModel,
  OLLAMA_DEFAULT_HOST,
} from '@google/gemini-cli-core';

interface OllamaModelDialogProps {
  ollamaHost?: string;
  onSelect: (model: string) => void;
  onCancel: () => void;
}

export function OllamaModelDialog({
  ollamaHost = OLLAMA_DEFAULT_HOST,
  onSelect,
  onCancel,
}: OllamaModelDialogProps): React.JSX.Element {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useInput((_input, key) => {
    if (key.escape) {
      onCancel();
    }
  });

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchOllamaModels(ollamaHost)
      .then((result: OllamaModel[]) => {
        if (!cancelled) {
          setModels(result);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : String(err);
          setError(
            `Could not connect to Ollama at ${ollamaHost}.\n` +
              `Make sure Ollama is running: ollama serve\n` +
              `Or configure a different host via OLLAMA_HOST.\n\n` +
              `Error: ${msg}`,
          );
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [ollamaHost]);

  const items = models.map((m) => {
    const size = m.size ? ` (${(m.size / 1_000_000_000).toFixed(1)} GB)` : '';
    const family = m.details?.family ? ` [${m.details.family}]` : '';
    return {
      label: `${m.name}${family}${size}`,
      value: m.name,
      key: m.name,
    };
  });

  return (
    <Box
      borderStyle="round"
      borderColor={theme.ui.focus}
      flexDirection="column"
      padding={1}
      width="100%"
    >
      <Text bold color={theme.text.primary}>
        Select Ollama Model
      </Text>
      <Box marginTop={1}>
        <Text color={theme.text.secondary}>
          Ollama server: <Text color={theme.text.link}>{ollamaHost}</Text>
        </Text>
      </Box>

      {loading && (
        <Box marginTop={1}>
          <Text color={theme.text.secondary}>Fetching available models…</Text>
        </Box>
      )}

      {error && (
        <Box marginTop={1} flexDirection="column">
          <Text color={theme.status.error}>{error}</Text>
          <Box marginTop={1}>
            <Text color={theme.text.secondary}>
              (Press Esc to go back and choose a different auth method)
            </Text>
          </Box>
        </Box>
      )}

      {!loading && !error && items.length === 0 && (
        <Box marginTop={1} flexDirection="column">
          <Text color={theme.status.warning}>
            No models found on the Ollama server.
          </Text>
          <Text color={theme.text.secondary}>
            Pull a model first: ollama pull llama3.1
          </Text>
          <Box marginTop={1}>
            <Text color={theme.text.secondary}>(Press Esc to go back)</Text>
          </Box>
        </Box>
      )}

      {!loading && !error && items.length > 0 && (
        <Box marginTop={1} flexDirection="column">
          <RadioButtonSelect
            items={items}
            initialIndex={0}
            onSelect={onSelect}
            isFocused={true}
          />
          <Box marginTop={1}>
            <Text color={theme.text.secondary}>
              (Use ↑↓ to navigate, Enter to select, Esc to cancel)
            </Text>
          </Box>
        </Box>
      )}
    </Box>
  );
}
