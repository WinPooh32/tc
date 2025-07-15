package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

func init() {
	log.SetFlags(0)
	log.SetOutput(io.Discard)
}

func main() {
	modelName := flag.String("model", "tiiuae/falcon-7b", "model name as at Huggingface model hub e.g. 'tiiuae/falcon-7b'")
	flag.Parse()

	configFile, err := tokenizer.CachedPath(*modelName, "tokenizer.json")
	if err != nil {
		fmt.Fprintln(os.Stderr, "get tokenizer:", err)
		os.Exit(1)
	}

	tk, err := pretrained.FromFile(configFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "make pretrained tokenizer:", err)
		os.Exit(1)
	}

	bs, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read input:", err)
		os.Exit(1)
	}

	enc, err := tk.EncodeSingle(string(bs), true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode:", err)
		os.Exit(1)
	}

	fmt.Println(len(enc.GetTokens()))
}
