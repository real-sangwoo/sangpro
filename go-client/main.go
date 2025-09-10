package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

type searchRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

type searchHit struct {
	ID       string                 `json:"id"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata"`
}

type searchResponse struct {
	Hits []searchHit `json:"hits"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: search-client \"your query here\"")
		os.Exit(1)
	}

	query := os.Args[1]

	reqBody := searchRequest{
		Query: query,
		TopK:  5,
	}

	buf, err := json.Marshal(reqBody)
	if err != nil {
		fmt.Println("failed to marshal request:", err)
		os.Exit(1)
	}

	resp, err := http.Post("http://localhost:8000/search", "application/json", bytes.NewReader(buf))
	if err != nil {
		fmt.Println("request error:", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Println("unexpected status:", resp.Status)
		os.Exit(1)
	}

	var sr searchResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		fmt.Println("decode error:", err)
		os.Exit(1)
	}

	for i, hit := range sr.Hits {
		fmt.Printf("%d. id=%s score=%.4f metadata=%v\n", i+1, hit.ID, hit.Score, hit.Metadata)
	}
}

