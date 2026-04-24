/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  OllamaContentGenerator,
  fetchOllamaModels,
  OLLAMA_DEFAULT_HOST,
} from './ollamaContentGenerator.js';
import { LlmRole } from '../telemetry/llmRole.js';

// Global fetch mock
const globalFetch = vi.fn();
vi.stubGlobal('fetch', globalFetch);

function makeOllamaChunk(
  text: string,
  done: boolean = false,
  promptEvalCount?: number,
  evalCount?: number,
) {
  return {
    model: 'llama3',
    created_at: '2024-01-01T00:00:00Z',
    message: { role: 'assistant', content: text },
    done,
    ...(done && {
      done_reason: 'stop',
      prompt_eval_count: promptEvalCount ?? 10,
      eval_count: evalCount ?? 5,
    }),
  };
}

describe('OllamaContentGenerator', () => {
  let generator: OllamaContentGenerator;

  beforeEach(() => {
    generator = new OllamaContentGenerator(OLLAMA_DEFAULT_HOST, 'llama3');
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should use default host and model', () => {
      const gen = new OllamaContentGenerator();
      expect(gen.getModel()).toBe('llama3');
    });

    it('should strip trailing slash from host', () => {
      const gen = new OllamaContentGenerator(
        'http://localhost:11434/',
        'llama3',
      );
      expect(gen.getModel()).toBe('llama3');
    });

    it('should use provided model', () => {
      const gen = new OllamaContentGenerator(OLLAMA_DEFAULT_HOST, 'mistral');
      expect(gen.getModel()).toBe('mistral');
    });
  });

  describe('generateContent', () => {
    it('should call /api/chat with stream: false', async () => {
      const chunk = makeOllamaChunk('Hello!', true, 10, 5);
      globalFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => chunk,
      });

      const result = await generator.generateContent(
        {
          model: 'llama3',
          contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
        },
        'prompt-id',
        LlmRole.MAIN,
      );

      expect(globalFetch).toHaveBeenCalledWith(
        `${OLLAMA_DEFAULT_HOST}/api/chat`,
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"stream":false'),
        }),
      );

      expect(result.candidates?.[0]?.content?.parts?.[0]).toEqual({
        text: 'Hello!',
      });
    });

    it('should throw on HTTP error with helpful message', async () => {
      globalFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        text: async () => 'model not found',
      });

      await expect(
        generator.generateContent(
          {
            model: 'llama3',
            contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
          },
          'prompt-id',
          LlmRole.MAIN,
        ),
      ).rejects.toThrow(/Ollama API error \(404\)/);
    });

    it('should include system instruction when provided', async () => {
      const chunk = makeOllamaChunk('Response', true);
      globalFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => chunk,
      });

      await generator.generateContent(
        {
          model: 'llama3',
          contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
          config: {
            systemInstruction: 'You are a helpful assistant.',
          },
        },
        'prompt-id',
        LlmRole.MAIN,
      );

      const callBody = JSON.parse(
        (globalFetch.mock.calls[0][1] as RequestInit).body as string,
      ) as { messages: Array<{ role: string; content: string }> };
      expect(callBody.messages[0]).toEqual({
        role: 'system',
        content: 'You are a helpful assistant.',
      });
    });

    it('should map model role to assistant in request', async () => {
      const chunk = makeOllamaChunk('Response', true);
      globalFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => chunk,
      });

      await generator.generateContent(
        {
          model: 'llama3',
          contents: [
            { role: 'user', parts: [{ text: 'Hi' }] },
            { role: 'model', parts: [{ text: 'Hello there' }] },
            { role: 'user', parts: [{ text: 'Continue' }] },
          ],
        },
        'prompt-id',
        LlmRole.MAIN,
      );

      const callBody = JSON.parse(
        (globalFetch.mock.calls[0][1] as RequestInit).body as string,
      ) as { messages: Array<{ role: string }> };
      expect(callBody.messages[1].role).toBe('assistant');
    });
  });

  describe('generateContentStream', () => {
    it('should call /api/chat with stream: true', async () => {
      const chunks = [
        makeOllamaChunk('Hel', false),
        makeOllamaChunk('lo!', true, 10, 5),
      ];
      const streamBody = chunks.map((c) => JSON.stringify(c)).join('\n');

      const mockReader = {
        read: vi
          .fn()
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode(streamBody),
          })
          .mockResolvedValueOnce({ done: true, value: undefined }),
        releaseLock: vi.fn(),
      };

      globalFetch.mockResolvedValueOnce({
        ok: true,
        body: { getReader: () => mockReader },
      });

      const streamGen = await generator.generateContentStream(
        {
          model: 'llama3',
          contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
        },
        'prompt-id',
        LlmRole.MAIN,
      );

      const responses = [];
      for await (const response of streamGen) {
        responses.push(response);
      }

      expect(responses).toHaveLength(2);
      expect(responses[0].candidates?.[0]?.content?.parts?.[0]).toEqual({
        text: 'Hel',
      });
      expect(responses[1].candidates?.[0]?.content?.parts?.[0]).toEqual({
        text: 'lo!',
      });
    });
  });

  describe('countTokens', () => {
    it('should estimate token count as chars/4', async () => {
      const result = await generator.countTokens({
        model: 'llama3',
        contents: [{ role: 'user', parts: [{ text: 'Hello world' }] }],
      });
      // "Hello world" = 11 chars => ceil(11/4) = 3
      expect(result.totalTokens).toBe(3);
    });

    it('should return 0 for empty contents', async () => {
      const result = await generator.countTokens({
        model: 'llama3',
        contents: [],
      });
      expect(result.totalTokens).toBe(0);
    });
  });

  describe('embedContent', () => {
    it('should call /api/embed and return embeddings', async () => {
      globalFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ embeddings: [[0.1, 0.2, 0.3]] }),
      });

      const result = await generator.embedContent({
        contents: [{ role: 'user', parts: [{ text: 'embed this' }] }],
        model: 'llama3',
      });

      expect(globalFetch).toHaveBeenCalledWith(
        `${OLLAMA_DEFAULT_HOST}/api/embed`,
        expect.any(Object),
      );
      expect(result.embeddings?.[0]?.values).toEqual([0.1, 0.2, 0.3]);
    });

    it('should throw on embed API error', async () => {
      globalFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'internal server error',
      });

      await expect(
        generator.embedContent({
          contents: [{ role: 'user', parts: [{ text: 'embed' }] }],
          model: 'llama3',
        }),
      ).rejects.toThrow(/Ollama embed API error/);
    });
  });
});

describe('fetchOllamaModels', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch models from /api/tags', async () => {
    const mockModels = [
      {
        name: 'llama3:latest',
        model: 'llama3:latest',
        modified_at: '2024-01-01',
        size: 4_700_000_000,
        digest: 'abc123',
        details: { family: 'llama', parameter_size: '8B' },
      },
    ];

    globalFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: mockModels }),
    });

    const models = await fetchOllamaModels();

    expect(globalFetch).toHaveBeenCalledWith(
      `${OLLAMA_DEFAULT_HOST}/api/tags`,
      expect.any(Object),
    );
    expect(models).toHaveLength(1);
    expect(models[0].name).toBe('llama3:latest');
  });

  it('should throw when server is unreachable', async () => {
    globalFetch.mockResolvedValueOnce({
      ok: false,
      status: 0,
      statusText: 'Connection refused',
    });

    await expect(fetchOllamaModels()).rejects.toThrow(
      /Failed to fetch Ollama models/,
    );
  });

  it('should use custom host', async () => {
    globalFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [] }),
    });

    await fetchOllamaModels('http://my-ollama:8080');

    expect(globalFetch).toHaveBeenCalledWith(
      'http://my-ollama:8080/api/tags',
      expect.any(Object),
    );
  });
});
