import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EvaluationPage } from "../pages/EvaluationPage";
import { ModelsPage } from "../pages/ModelsPage";

vi.mock("../api", () => ({
  api: {
    models: vi.fn(),
    evaluation: vi.fn(),
    stats: vi.fn(),
  },
}));
import { api } from "../api";

describe("Models and Evaluation pages", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("explains unavailable models without presenting configuration as a trained checkpoint", async () => {
    (api.models as ReturnType<typeof vi.fn>).mockResolvedValue({
      default_model: null,
      models: [
        {
          name: "gru_glove",
          type: "gru",
          embedding_dim: 1024,
          available: false,
          unavailable_reason: "missing model file: best.pt",
          num_images: 0,
          index_backend: null,
          metadata: {},
        },
      ],
    });

    render(<ModelsPage />);

    expect(await screen.findByText("missing model file: best.pt")).toBeInTheDocument();
    expect(screen.getByText("unavailable")).toBeInTheDocument();
  });

  it("shows evaluation as pending when no measured results exist", async () => {
    (api.evaluation as ReturnType<typeof vi.fn>).mockResolvedValue({
      available: false,
      message: "No evaluation results found yet.",
      results: [],
      histories: {},
      reference: { model: "GRU + GloVe", "t2i_R@1": 37, "t2i_R@10": 86.8, note: "reference only" },
      synthetic_warning: false,
    });
    (api.stats as ReturnType<typeof vi.fn>).mockResolvedValue({
      available: false,
      message: "Dataset statistics not found.",
      dataset: {},
      stats: {},
      indexed_images: {},
      synthetic: false,
    });

    render(<EvaluationPage />);

    expect(await screen.findByText("No evaluation results yet")).toBeInTheDocument();
    expect(screen.getByText("Reference values from the paper (not measured here)")).toBeInTheDocument();
    expect(screen.getByText("Dataset statistics not found.")).toBeInTheDocument();
  });

  it("renders only saved measured evaluation rows", async () => {
    (api.evaluation as ReturnType<typeof vi.fn>).mockResolvedValue({
      available: true,
      message: null,
      results: [
        {
          name: "gru_glove",
          model_type: "gru",
          split: "test",
          protocol: "full",
          metrics: { "t2i_R@1": 12.5, "t2i_R@5": 40, "t2i_R@10": 60, "t2i_mean_rank": 20, "i2t_R@1": 10, "i2t_R@10": 55 },
          num_images: 1000,
          num_captions: 5000,
          embedding_dim: 1024,
          latency_ms_median: 2.4,
          latency_setup: "synthetic test",
          training_seconds: 12,
          dataset: "MS-COCO",
          is_subset: false,
          synthetic: false,
          file: "gru_glove_test_full.json",
        },
      ],
      histories: {},
      reference: {},
      synthetic_warning: false,
    });
    (api.stats as ReturnType<typeof vi.fn>).mockResolvedValue({
      available: false,
      message: "No statistics.",
      dataset: {},
      stats: {},
      indexed_images: {},
      synthetic: false,
    });

    render(<EvaluationPage />);

    const row = await screen.findByRole("row", { name: /gru_glove/ });
    expect(row).toHaveTextContent("12.5");
    expect(row).toHaveTextContent("1000");
  });
});
