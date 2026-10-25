package chat

import (
	"testing"

	"github.com/Mini-llm-lab/Mini-CLI/internal/inference"
)

type recordingGenerator struct {
	calls [][]inference.Message
}

func (r *recordingGenerator) Generate(messages []inference.Message, _ inference.GenerationOptions) (string, error) {
	r.calls = append(r.calls, messages)
	return "answer", nil
}

func (r *recordingGenerator) Close() error { return nil }

func TestSendSendsTranscript(t *testing.T) {
	g := &recordingGenerator{}
	c := New(g, inference.DefaultOptions())
	for _, q := range []string{"first", "second"} {
		if _, err := c.Send(q); err != nil {
			t.Fatal(err)
		}
	}
	if len(g.calls) != 2 {
		t.Fatalf("expected 2 generate calls, got %d", len(g.calls))
	}
	want := []inference.Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "answer"},
		{Role: "user", Content: "second"},
	}
	if len(g.calls[1]) != len(want) {
		t.Fatalf("second call must carry the transcript (the engine prunes it), got %+v", g.calls[1])
	}
	for i, m := range want {
		if g.calls[1][i].Role != m.Role || g.calls[1][i].Content != m.Content {
			t.Fatalf("message %d: got %+v, want %+v", i, g.calls[1][i], m)
		}
	}
	if len(c.History) != 4 || c.History[2].Content != "second" || c.History[3].Content != "answer" {
		t.Fatalf("history should stay a transcript, got %+v", c.History)
	}
}

func TestSendAppendsHistory(t *testing.T) {
	c := New(&inference.MockGenerator{}, inference.DefaultOptions())
	ans, err := c.Send("hello")
	if err != nil {
		t.Fatal(err)
	}
	if ans == "" || len(c.History) != 2 {
		t.Fatalf("got %q %+v", ans, c.History)
	}
	if c.History[0].Role != "user" || c.History[1].Role != "assistant" {
		t.Fatalf("%+v", c.History)
	}
	c.Reset()
	if len(c.History) != 0 {
		t.Fatal("reset failed")
	}
}

func TestSendEmptyRejected(t *testing.T) {
	c := New(&inference.MockGenerator{}, inference.DefaultOptions())
	if _, err := c.Send("   "); err == nil {
		t.Fatal("expected error")
	}
}
