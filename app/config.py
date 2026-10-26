from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: str = ""
    tavily_api_key: str = ""
    llm_model: str = "openai/gpt-4o-mini"
    research_model: str | None = None
    writer_model: str | None = None
    fake_pipeline: bool = False
    cors_origin: str = "http://localhost:5173"

    @property
    def researcher_model(self) -> str:
        return self.research_model or self.llm_model

    @property
    def writer_llm_model(self) -> str:
        return self.writer_model or self.llm_model


settings = Settings()
