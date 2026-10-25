package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/Mini-llm-lab/Mini-CLI/internal/chat"
	"github.com/Mini-llm-lab/Mini-CLI/internal/inference"
)

type answerMsg struct {
	text string
	err  error
}

type Model struct {
	title    string
	conv     *chat.Conversation
	input    textinput.Model
	viewport viewport.Model
	spinner  spinner.Model
	loading  bool
	quitting bool
	errMsg   string
}

func New(title string, conv *chat.Conversation) Model {
	ti := textinput.New()
	ti.Placeholder = "Ask Mini…  (/reset, /quit)"
	ti.Focus()
	ti.CharLimit = 2000

	vp := viewport.New(80, 20)

	sp := spinner.New()
	sp.Spinner = spinner.Dot

	return Model{title: title, conv: conv, input: ti, viewport: vp, spinner: sp}
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(textinput.Blink, m.spinner.Tick)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.viewport.Width = msg.Width - 4
		m.viewport.Height = msg.Height - 6
		m.render()
		return m, nil
	case answerMsg:
		m.loading = false
		if msg.err != nil {
			m.errMsg = msg.err.Error()
		} else {
			m.errMsg = ""
		}
		m.render()
		return m, nil
	case spinner.TickMsg:
		if m.loading {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			m.render()
			return m, cmd
		}
		return m, nil
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			m.quitting = true
			return m, tea.Quit
		case tea.KeyEnter:
			if m.loading {
				return m, nil
			}
			q := strings.TrimSpace(m.input.Value())
			m.input.SetValue("")
			if q == "" {
				return m, nil
			}
			lower := strings.ToLower(q)
			if lower == "/quit" || lower == "/exit" {
				m.quitting = true
				return m, tea.Quit
			}
			if lower == "/reset" {
				m.conv.Reset()
				m.errMsg = ""
				m.render()
				return m, nil
			}
			m.conv.History = append(m.conv.History, inference.Message{Role: "user", Content: q})
			m.loading = true
			m.render()
			opts := m.conv.Options
			gen := m.conv.Generator
			history := append([]inference.Message(nil), m.conv.History...)
			return m, tea.Batch(m.spinner.Tick, func() tea.Msg {
				text, err := gen.Generate(history, opts)
				if err == nil {
					m.conv.History = append(m.conv.History, inference.Message{Role: "assistant", Content: strings.TrimSpace(text)})
				} else {
					m.conv.History = m.conv.History[:len(m.conv.History)-1]
				}
				return answerMsg{text: text, err: err}
			})
		}
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m *Model) render() {
	width := m.viewport.Width
	if width <= 0 {
		width = 80
	}
	var b strings.Builder
	for _, msg := range m.conv.History {
		if msg.Role == "user" {
			b.WriteString(wrapMessage(UserStyle.Render("You > "), msg.Content, width) + "\n\n")
		} else {
			b.WriteString(wrapMessage(AIStyle.Render("AI  > "), msg.Content, width) + "\n\n")
		}
	}
	if m.loading {
		b.WriteString(m.spinner.View() + " generating…\n")
	}
	if m.errMsg != "" {
		b.WriteString(wordWrap(ErrorStyle.Render("Error: "+m.errMsg), width) + "\n")
	}
	m.viewport.SetContent(b.String())
	m.viewport.GotoBottom()
}

func (m Model) View() string {
	if m.quitting {
		return "Bye.\n"
	}
	header := TitleStyle.Render(m.title)
	hint := HintStyle.Render("Enter send • /reset clear • /quit or Ctrl+C exit")
	return fmt.Sprintf("%s\n%s\n%s\n%s", header, m.viewport.View(), m.input.View(), hint)
}
