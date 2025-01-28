"""Module for setting up RAG movie recommendation crew and agents."""

import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import EXASearchTool

from rag_movie_agent.tools.chroma_vector_search_tool import ChromaVectorSearchTool

# Initialize tools
chroma_vector_search_tool = ChromaVectorSearchTool(
    collection_name="netflix_data",
    limit=10,
)

exa_search_tool = EXASearchTool(
    api_key=os.environ.get("EXA_API_KEY"),
)

@CrewBase
class MovieRecommendationCrew:
    """Crew for movie recommendations using RAG."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def rag_agent(self) -> Agent:
        """Create agent for RAG retrieval."""
        return Agent(
            config=self.agents_config["rag_agent"],
            verbose=True,
            tools=[chroma_vector_search_tool],
        )

    @agent 
    def expand_details_agent(self) -> Agent:
        """Create agent for expanding movie details."""
        return Agent(
            config=self.agents_config["expand_details_agent"],
            verbose=True, 
            tools=[exa_search_tool],
        )

    @agent
    def report_agent(self) -> Agent:
        """Create agent for generating final report."""
        return Agent(
            config=self.agents_config["report_agent"],
            verbose=True
        )

    @task
    def rag_task(self) -> Task:
        """Define RAG retrieval task."""
        return Task(
            config=self.tasks_config["rag_task"],
        )

    @task 
    def expand_details_task(self) -> Task:
        """Define details expansion task."""
        return Task(
            config=self.tasks_config["expand_details_task"],
            output_file="report.md"
        )

    @task
    def report_task(self) -> Task:
        """Define report generation task."""
        return Task(
            config=self.tasks_config["report_task"],
            output_file="report.md"
        )

    @crew
    def crew(self) -> Crew:
        """Create the movie recommendation crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
