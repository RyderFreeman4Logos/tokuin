/// Conservative tokenizer: returns the higher of OpenAI BPE and chars/3 estimates.
///
/// Claude models use a ~65K vocabulary (vs OpenAI's 200K o200k_base), producing
/// more tokens for the same text. Since Anthropic does not publish their tokenizer,
/// chars/3 serves as a conservative approximation for Claude-class models.
/// Taking max(openai, chars/3) ensures we never undercount for either family.
#[cfg(feature = "conservative")]
use crate::error::TokenizerError;
use crate::tokenizers::openai::OpenAITokenizer;
use crate::tokenizers::Tokenizer;

pub struct ConservativeTokenizer {
    openai: OpenAITokenizer,
}

impl ConservativeTokenizer {
    pub fn new(openai_model: &str) -> Result<Self, TokenizerError> {
        Ok(Self {
            openai: OpenAITokenizer::new(openai_model)?,
        })
    }

    fn chars_div3_count(text: &str) -> usize {
        let char_count = text.chars().count();
        (char_count as f64 / 3.0).ceil() as usize
    }
}

impl Tokenizer for ConservativeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        self.openai.encode(text)
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, TokenizerError> {
        self.openai.decode(tokens)
    }

    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        let openai_count = self.openai.count_tokens(text)?;
        let approx_count = Self::chars_div3_count(text);
        Ok(openai_count.max(approx_count))
    }

    fn name(&self) -> &str {
        "conservative"
    }

    fn input_price_per_1k(&self) -> Option<f64> {
        None
    }

    fn output_price_per_1k(&self) -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservative_always_gte_openai() {
        let conservative = ConservativeTokenizer::new("gpt-4o").unwrap();
        let openai = OpenAITokenizer::new("gpt-4o").unwrap();
        let texts = [
            "Hello, world!",
            "fn main() { println!(\"test\"); }",
            "这是中文文本测试",
            "a]b[c{d}e(f)g",
            &"x".repeat(10000),
        ];
        for text in &texts {
            let c = conservative.count_tokens(text).unwrap();
            let o = openai.count_tokens(text).unwrap();
            assert!(c >= o, "conservative ({c}) < openai ({o}) for: {text:.40}");
        }
    }

    #[test]
    fn test_conservative_always_gte_chars_div3() {
        let conservative = ConservativeTokenizer::new("gpt-4o").unwrap();
        let texts = ["Hello", "fn main() {}", "中文", &"abc".repeat(1000)];
        for text in &texts {
            let c = conservative.count_tokens(text).unwrap();
            let approx = (text.chars().count() as f64 / 3.0).ceil() as usize;
            assert!(c >= approx, "conservative ({c}) < approx ({approx})");
        }
    }

    #[test]
    fn test_empty_string() {
        let conservative = ConservativeTokenizer::new("gpt-4o").unwrap();
        assert_eq!(conservative.count_tokens("").unwrap(), 0);
    }
}
