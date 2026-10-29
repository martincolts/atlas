package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const defaultBaseURL = "https://api.anthropic.com"

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Request struct {
	Model     string    `json:"model"`
	MaxTokens int       `json:"max_tokens"`
	System    string    `json:"system,omitempty"`
	Messages  []Message `json:"messages"`
	Stream    bool      `json:"stream"`
}

type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// Result es lo que medimos de cada request.
type Result struct {
	Text       string
	StopReason string
	Usage      Usage
	TTFT       time.Duration // hasta el primer token de texto
	Total      time.Duration
}

// TokensPerSecond mide solo la fase de decode.
func (r Result) TokensPerSecond() float64 {
	decode := r.Total - r.TTFT
	if decode <= 0 || r.Usage.OutputTokens == 0 {
		return 0
	}
	return float64(r.Usage.OutputTokens) / decode.Seconds()
}

type Client struct {
	APIKey  string
	BaseURL string // vacío = API de Anthropic; en la semana 2, tu gateway
	HTTP    *http.Client
}

// APIError conserva el status para decidir si reintentar.
type APIError struct {
	Status     int
	RetryAfter string
	Body       string
}

func (e *APIError) Error() string { return fmt.Sprintf("api error %d: %s", e.Status, e.Body) }

// streamEvent cubre los campos que usamos de los eventos SSE.
type streamEvent struct {
	Type    string `json:"type"`
	Message struct {
		Usage Usage `json:"usage"`
	} `json:"message"`
	Delta struct {
		Type       string `json:"type"`
		Text       string `json:"text"`
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage Usage `json:"usage"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func (c *Client) baseURL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return defaultBaseURL
}

func (c *Client) Stream(ctx context.Context, req Request, onText func(string)) (Result, error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return Result{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL()+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return Result{}, err
	}
	httpReq.Header.Set("x-api-key", c.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("content-type", "application/json")

	start := time.Now()
	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return Result{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return Result{}, &APIError{Status: resp.StatusCode, RetryAfter: resp.Header.Get("retry-after"), Body: string(b)}
	}

	var res Result
	var sb strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue // ignoramos "event:" y líneas vacías
		}
		var ev streamEvent
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev); err != nil {
			return res, fmt.Errorf("evento inválido: %w", err)
		}
		switch ev.Type {
		case "message_start":
			res.Usage = ev.Message.Usage // tokens de entrada y de cache
		case "content_block_delta":
			if ev.Delta.Type == "text_delta" {
				if res.TTFT == 0 {
					res.TTFT = time.Since(start)
				}
				sb.WriteString(ev.Delta.Text)
				if onText != nil {
					onText(ev.Delta.Text)
				}
			}
		case "message_delta":
			res.StopReason = ev.Delta.StopReason
			res.Usage.OutputTokens = ev.Usage.OutputTokens
		case "error":
			// un error a mitad del stream: ya recibimos un 200
			return res, fmt.Errorf("error en el stream: %s: %s", ev.Error.Type, ev.Error.Message)
		}
	}
	if err := sc.Err(); err != nil {
		return res, err
	}
	res.Text = sb.String()
	res.Total = time.Since(start)
	return res, nil
}