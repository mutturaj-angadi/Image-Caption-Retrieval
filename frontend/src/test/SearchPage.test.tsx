import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api";
import { SearchPage } from "../pages/SearchPage";
import type { ModelsResponse, SearchResponse } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, api: { models: vi.fn(), search: vi.fn() } };
});
import { api } from "../api";

const MODELS: ModelsResponse = {
  default_model: "gru_glove",
  models: [
    {
      name: "gru_glove",
      type: "gru",
      embedding_dim: 1024,
      available: true,
      unavailable_reason: null,
      num_images: 5000,
      index_backend: "exact",
      metadata: {},
    },
  ],
};

const RESPONSE: SearchResponse = {
  query: "a dog in snow",
  model: "gru_glove",
  top_k: 10,
  num_results: 2,
  unknown_tokens: ["snowy"],
  score_type: "cosine_similarity",
  score_note: "Cosine similarity ...",
  latency_ms: 3.2,
  results: [
    { rank: 1, image_id: "11", score: 0.71, image_url: "/api/images/11", file_name: "a.jpg", captions: ["A dog runs in snow"], split: "test", image_available: true },
    { rank: 2, image_id: "12", score: 0.55, image_url: "/api/images/12", file_name: "b.jpg", captions: [], split: "test", image_available: false },
  ],
};

describe("SearchPage", () => {
  beforeEach(() => {
    (api.models as ReturnType<typeof vi.fn>).mockResolvedValue(MODELS);
    (api.search as ReturnType<typeof vi.fn>).mockReset();
  });

  it("asks for a description instead of calling the API when the query is blank", async () => {
    render(<SearchPage />);
    await userEvent.click(screen.getByRole("button", { name: /^search$/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/describe the image/i);
    expect(api.search).not.toHaveBeenCalled();
  });

  it("shows ranked results with scores, captions, unknown words and an unavailable-image placeholder", async () => {
    (api.search as ReturnType<typeof vi.fn>).mockResolvedValue(RESPONSE);
    render(<SearchPage />);
    await userEvent.type(screen.getByLabelText(/describe the image/i), "a dog in snow");
    await userEvent.click(screen.getByRole("button", { name: /^search$/i }));
    expect(await screen.findByText(/2 results for/i)).toBeInTheDocument();
    expect(api.search).toHaveBeenCalledWith("a dog in snow", 10, undefined, expect.any(AbortSignal));
    expect(screen.getByText("0.710")).toBeInTheDocument();
    expect(screen.getByText("A dog runs in snow")).toBeInTheDocument();
    expect(screen.getByText(/snowy/)).toBeInTheDocument();
    expect(screen.getByText(/image file unavailable/i)).toBeInTheDocument();
  });

  it("shows the backend error message when the search fails", async () => {
    (api.search as ReturnType<typeof vi.fn>).mockRejectedValue(new ApiError(503, "model_unavailable", "model 'gru_glove' is not available"));
    render(<SearchPage />);
    await userEvent.type(screen.getByLabelText(/describe the image/i), "a dog");
    await userEvent.click(screen.getByRole("button", { name: /^search$/i }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/not available/i));
    expect(screen.queryByText(/results for/i)).not.toBeInTheDocument();
  });

  it("opens the lightbox with all captions and closes it with Escape", async () => {
    (api.search as ReturnType<typeof vi.fn>).mockResolvedValue(RESPONSE);
    render(<SearchPage />);
    await userEvent.type(screen.getByLabelText(/describe the image/i), "a dog in snow");
    await userEvent.click(screen.getByRole("button", { name: /^search$/i }));
    await userEvent.click(await screen.findByRole("button", { name: /open result 1/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("warns when no model is available", async () => {
    (api.models as ReturnType<typeof vi.fn>).mockResolvedValue({
      default_model: null,
      models: [{ ...MODELS.models[0], available: false, unavailable_reason: "missing best.pt" }],
    });
    render(<SearchPage />);
    expect(await screen.findByText(/no model is available yet/i)).toBeInTheDocument();
    expect(screen.getByText(/missing best.pt/)).toBeInTheDocument();
  });
});
