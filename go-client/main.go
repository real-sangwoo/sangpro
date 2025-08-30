package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

type queryPayload struct {
	Text  string `json:"query"`
	Limit int    `json:"top_k"`
}

type resultItem struct {
	DocumentID string                 `json:"id"`
	Relevance  float64                `json:"score"`
	Attrs      map[string]interface{} `json:"metadata"`
}

type queryResult struct {
	Items []resultItem `json:"hits"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: search-client \"your query here\"")
		os.Exit(1)
	}

	userQuery := os.Args[1]

	payload := queryPayload{
		Text:  userQuery,
		Limit: 5,
	}

	encoded, encErr := json.Marshal(payload)
	if encErr != nil {
		fmt.Println("failed to marshal request:", encErr)
		os.Exit(1)
	}

	httpResp, httpErr := http.Post("http://localhost:8000/search", "application/json", bytes.NewReader(encoded))
	if httpErr != nil {
		fmt.Println("request error:", httpErr)
		os.Exit(1)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		fmt.Println("unexpected status:", httpResp.Status)
		os.Exit(1)
	}

	var parsed queryResult
	if decErr := json.NewDecoder(httpResp.Body).Decode(&parsed); decErr != nil {
		fmt.Println("decode error:", decErr)
		os.Exit(1)
	}

	for n, item := range parsed.Items {
		fmt.Printf("%d. id=%s score=%.4f metadata=%v\n", n+1, item.DocumentID, item.Relevance, item.Attrs)
	}
}
