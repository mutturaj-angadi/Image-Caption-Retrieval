import { describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api";

function mockFetch(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: status >= 200 && status < 300, status, json: async () => body }),
  );
}

describe("api client", () => {
  it("posts the search request as JSON", async () => {
    mockFetch(200, { results: [] });
    await api.search("a red circle", 5, "m1");
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("/api/search");
    expect(JSON.parse(init.body)).toEqual({ query: "a red circle", top_k: 5, model: "m1" });
  });

  it("forwards an AbortSignal so callers can cancel pending requests", async () => {
    mockFetch(200, { results: [] });
    const controller = new AbortController();
    await api.search("a red circle", 5, undefined, controller.signal);
    const [, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(init.signal).toBe(controller.signal);
  });

  it("maps the backend error shape to ApiError", async () => {
    mockFetch(422, { error: { code: "invalid_query", message: "query must not be empty" } });
    await expect(api.search("", 5)).rejects.toMatchObject({ status: 422, code: "invalid_query", message: "query must not be empty" });
  });

  it("uses a generic message when the error body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 502, json: async () => Promise.reject(new Error("no json")) }));
    await expect(api.health()).rejects.toMatchObject({ status: 502, code: "http_error" });
  });

  it("reports a network failure clearly", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const err = await api.models().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("network_error");
    expect(err.message).toMatch(/backend/i);
  });
});
