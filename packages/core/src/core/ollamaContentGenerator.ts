/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Content,
  ContentListUnion,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
  Part,
  Tool,
} from '@google/genai';
import {
  FinishReason,
  GenerateContentResponse as GCResponse,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

/** Default Ollama server URL */
export const OLLAMA_DEFAULT_HOST = 'http://localhost:11434';

/** A single model returned from Ollama's /api/tags endpoint. */
export interface OllamaModel {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details?: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
  };
}

/** Shape of the /api/tags response */
interface OllamaTagsResponse {
  models: OllamaModel[];
}

/** Shape of an Ollama chat message */
interface OllamaMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  tool_calls?: OllamaToolCall[];
  images?: string[];
}

/** Ollama tool call format */
interface OllamaToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

/** Ollama tool definition format */
interface OllamaTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: {
      type: string;
      properties?: Record<string, unknown>;
      required?: string[];
    };
  };
}

/** Shape of a streaming chunk from /api/chat */
interface OllamaChatChunk {
  model: string;
  created_at: string;
  message: {
    role: 'assistant';
    content: string;
    tool_calls?: OllamaToolCall[];
  };
  done: boolean;
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
}

/**
 * Convert a single Content or Part item to an Ollama message.
 */
function contentItemToOllamaMessages(
  item: Content | Part | string,
): OllamaMessage | OllamaMessage[] {
  if (typeof item === 'string') {
    return { role: 'user', content: item };
  }
  // Treat items without a role as bare Parts (text-only user message)
  if (!('role' in item) || typeof item.role !== 'string') {
    const text =
      'text' in item && typeof item.text === 'string' ? item.text : '';
    return { role: 'user', content: text };
  }
  const content = item;
  const role =
    content.role === 'model' ? 'assistant' : (content.role ?? 'user');
  let text = '';
  const toolCalls: OllamaToolCall[] = [];
  const result: OllamaMessage[] = [];

  for (const part of content.parts ?? []) {
    if (part.text !== undefined) {
      text += part.text;
    } else if (part.functionCall) {
      const fc = part.functionCall;
      toolCalls.push({
        function: {
          name: fc.name ?? '',
          arguments: fc.args ?? {},
        },
      });
    } else if (part.functionResponse) {
      // For function responses, add as a tool message
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      const responseBody = part.functionResponse as unknown as {
        response?: unknown;
      };
      result.push({
        role: 'tool',
        content: JSON.stringify(responseBody.response ?? {}),
      });
    } else if (part.inlineData?.data) {
      result.push({
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        role: role as OllamaMessage['role'],
        content: text,
        images: [part.inlineData.data],
      });
      text = '';
    }
  }

  const msg: OllamaMessage = {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    role: role as OllamaMessage['role'],
    content: text,
  };
  if (toolCalls.length > 0) {
    msg.tool_calls = toolCalls;
  }
  result.push(msg);
  return result;
}

/**
 * Convert Gemini-style ContentListUnion to Ollama-style messages.
 */
function contentsToOllamaMessages(contents: ContentListUnion): OllamaMessage[] {
  const messages: OllamaMessage[] = [];
  const items = Array.isArray(contents) ? contents : [contents];
  for (const item of items) {
    const msgs = contentItemToOllamaMessages(item as Content | Part | string);
    if (Array.isArray(msgs)) {
      messages.push(...msgs);
    } else {
      messages.push(msgs);
    }
  }
  return messages;
}

/**
 * Convert Gemini FunctionDeclarations to Ollama tool definitions.
 */
function toolsToOllamaTools(
  tools: Tool[] | undefined,
): OllamaTool[] | undefined {
  if (!tools) return undefined;
  const result: OllamaTool[] = [];
  for (const tool of tools) {
    if (!tool.functionDeclarations) continue;
    for (const fd of tool.functionDeclarations) {
      result.push({
        type: 'function',
        function: {
          name: fd.name ?? '',
          description: fd.description ?? '',
          // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          parameters: fd.parameters as OllamaTool['function']['parameters'],
        },
      });
    }
  }
  return result.length > 0 ? result : undefined;
}

/**
 * Convert an OllamaChatChunk into a GenerateContentResponse.
 */
function chunkToResponse(chunk: OllamaChatChunk): GenerateContentResponse {
  const parts: Part[] = [];

  if (chunk.message.content) {
    parts.push({ text: chunk.message.content });
  }

  // Handle tool calls from the model
  if (chunk.message.tool_calls) {
    for (const tc of chunk.message.tool_calls) {
      parts.push({
        functionCall: {
          name: tc.function.name,
          args: tc.function.arguments,
        },
      });
    }
  }

  const responseData = {
    candidates: [
      {
        content: {
          role: 'model',
          parts,
        },
        index: 0,
        finishReason: chunk.done ? FinishReason.STOP : undefined,
      },
    ],
    usageMetadata: chunk.done
      ? {
          promptTokenCount: chunk.prompt_eval_count ?? 0,
          candidatesTokenCount: chunk.eval_count ?? 0,
          totalTokenCount:
            (chunk.prompt_eval_count ?? 0) + (chunk.eval_count ?? 0),
        }
      : undefined,
  };

  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  return Object.setPrototypeOf(responseData, GCResponse.prototype);
}

/**
 * Fetch available models from the Ollama server.
 */
export async function fetchOllamaModels(
  host: string = OLLAMA_DEFAULT_HOST,
): Promise<OllamaModel[]> {
  const url = `${host.replace(/\/$/, '')}/api/tags`;
  const response = await fetch(url, {
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) {
    throw new Error(
      `Failed to fetch Ollama models from ${url}: ${response.status} ${response.statusText}`,
    );
  }
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
  const data = (await response.json()) as OllamaTagsResponse;
  return data.models ?? [];
}

/**
 * ContentGenerator implementation for Ollama.
 *
 * Maps Gemini-style ContentGenerator API calls to Ollama's /api/chat endpoint.
 */
/**
 * Extract text from a systemInstruction value.
 */
function extractSystemText(si: unknown): string {
  if (typeof si === 'string') return si;
  if (si && typeof si === 'object' && 'parts' in si) {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const parts = (si as { parts?: Array<{ text?: string }> }).parts;
    return parts?.map((p) => p.text ?? '').join('') ?? '';
  }
  return '';
}

export class OllamaContentGenerator implements ContentGenerator {
  private readonly host: string;
  private model: string;

  constructor(host: string = OLLAMA_DEFAULT_HOST, model: string = 'llama3') {
    this.host = host.replace(/\/$/, '');
    this.model = model;
  }

  getModel(): string {
    return this.model;
  }

  setModel(model: string): void {
    this.model = model;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const messages = contentsToOllamaMessages(request.contents);

    // Prepend system instruction if provided
    if (request.config?.systemInstruction) {
      const systemText = extractSystemText(request.config.systemInstruction);
      if (systemText) {
        messages.unshift({ role: 'system', content: systemText });
      }
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const configTools = request.config?.['tools'] as Tool[] | undefined;
    const ollamaTools = toolsToOllamaTools(configTools);

    const body: Record<string, unknown> = {
      model: this.model,
      messages,
      stream: false,
    };

    if (ollamaTools && ollamaTools.length > 0) {
      body['tools'] = ollamaTools;
    }

    const response = await fetch(`${this.host}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Ollama API error (${response.status}): ${errorText}. ` +
          `Make sure Ollama is running at ${this.host} (ollama serve) ` +
          `and you have pulled the model (ollama pull ${this.model}).`,
      );
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const chunk = (await response.json()) as OllamaChatChunk;
    return chunkToResponse(chunk);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = contentsToOllamaMessages(request.contents);

    // Prepend system instruction if provided
    if (request.config?.systemInstruction) {
      const systemText = extractSystemText(request.config.systemInstruction);
      if (systemText) {
        messages.unshift({ role: 'system', content: systemText });
      }
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const configTools = request.config?.['tools'] as Tool[] | undefined;
    const ollamaTools = toolsToOllamaTools(configTools);

    const body: Record<string, unknown> = {
      model: this.model,
      messages,
      stream: true,
    };

    if (ollamaTools && ollamaTools.length > 0) {
      body['tools'] = ollamaTools;
    }

    const response = await fetch(`${this.host}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Ollama API error (${response.status}): ${errorText}. ` +
          `Make sure Ollama is running at ${this.host} (ollama serve) ` +
          `and you have pulled the model (ollama pull ${this.model}).`,
      );
    }

    const body_ = response.body;
    if (!body_) {
      throw new Error('Ollama API returned an empty response body.');
    }

    return this.streamBodyChunks(body_);
  }

  private async *streamBodyChunks(
    body_: ReadableStream<Uint8Array>,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = body_.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (value) {
          buffer += decoder.decode(value, { stream: true });
        }

        if (done) {
          // Flush remaining buffer
          const remaining = buffer.trim();
          if (remaining) {
            try {
              // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
              const chunk = JSON.parse(remaining) as OllamaChatChunk;
              if (chunk.model) {
                this.model = chunk.model;
              }
              yield chunkToResponse(chunk);
            } catch {
              // ignore malformed JSON
            }
          }
          break;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
            const chunk = JSON.parse(trimmed) as OllamaChatChunk;
            // Update model name if server returns it
            if (chunk.model) {
              this.model = chunk.model;
            }
            yield chunkToResponse(chunk);
            if (chunk.done) return;
          } catch {
            // ignore malformed JSON lines
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Ollama does not have a dedicated token counting endpoint.
    // Use a simple heuristic: ~4 chars per token.
    const rawContents = request.contents;
    const contentsArray = Array.isArray(rawContents)
      ? rawContents
      : rawContents
        ? [rawContents]
        : [];

    let totalChars = 0;
    for (const item of contentsArray) {
      if (typeof item === 'string') {
        totalChars += item.length;
      } else if ('parts' in item && Array.isArray(item.parts)) {
        for (const part of item.parts) {
          if (typeof part === 'object' && part.text !== undefined) {
            totalChars += part.text.length;
          }
        }
      }
    }

    const tokenCount = Math.ceil(totalChars / 4);
    return { totalTokens: tokenCount };
  }

  private extractTextForEmbed(item: unknown): string {
    if (typeof item === 'string') {
      return item;
    }
    if (
      item &&
      typeof item === 'object' &&
      'parts' in item &&
      Array.isArray(item.parts)
    ) {
      const parts: unknown[] = item.parts;
      return parts
        .map((p) => {
          if (typeof p !== 'object' || p === null || !('text' in p)) return '';
          const text = (p as { text?: unknown }).text;
          return typeof text === 'string' ? text : '';
        })
        .join('');
    }
    return '';
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const rawContents = request.contents;
    const firstItem = Array.isArray(rawContents) ? rawContents[0] : rawContents;

    const text = this.extractTextForEmbed(firstItem);
    const response = await fetch(`${this.host}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: text }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Ollama embed API error (${response.status}): ${errorText}`,
      );
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const data = (await response.json()) as { embeddings?: number[][] };
    const embeddings = data.embeddings?.[0] ?? [];

    return {
      embeddings: [{ values: embeddings }],
    };
  }
}
