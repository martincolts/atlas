package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/martincolts/atlas-bench/internal/llm"
)

// loadEnv sets variables from a KEY=value file without overriding ones
// already in the environment. A missing file is not an error.
func loadEnv(path string) error {
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(strings.TrimPrefix(line, "export "), "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.Trim(strings.TrimSpace(val), `"'`)
		if _, set := os.LookupEnv(key); !set {
			os.Setenv(key, val)
		}
	}
	return nil
}

func main() {
	if err := loadEnv(".env"); err != nil {
		log.Fatalf("loading .env: %v", err)
	}

	model := flag.String("model", os.Getenv("ATLAS_MODEL"), "ID del modelo")
	words := flag.Int("words", 100, "palabras de relleno en el prompt")
	maxTokens := flag.Int("max-tokens", 300, "tope de tokens de salida")
	runs := flag.Int("runs", 3, "repeticiones")
	flag.Parse()

	c := &llm.Client{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
		HTTP:   &http.Client{Timeout: 5 * time.Minute},
	}
	filler := strings.Repeat("Kafka replica cada partición en varios brokers. ", *words/7+1)
	prompt := "Contexto:\n" + filler + "\n\nResumí el contexto en tres oraciones."

	fmt.Println("run,input_tokens,output_tokens,ttft_ms,total_ms,tokens_per_s,stop_reason")
	for i := 1; i <= *runs; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		res, err := c.Stream(ctx, llm.Request{
			Model:     *model,
			MaxTokens: *maxTokens,
			Messages:  []llm.Message{{Role: "user", Content: prompt}},
		}, func(text string) {
			fmt.Fprint(os.Stderr, text)
		})
		fmt.Fprintln(os.Stderr)
		cancel()
		if err != nil {
			log.Fatalf("run %d: %v", i, err)
		}
		fmt.Printf("%d,%d,%d,%d,%d,%.1f,%s\n", i,
			res.Usage.InputTokens, res.Usage.OutputTokens,
			res.TTFT.Milliseconds(), res.Total.Milliseconds(),
			res.TokensPerSecond(), res.StopReason)
	}
}