from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from deepfix.task_domain.models import TaskLifecycleStatus
from task_domain.test_service_integration import config as config_fixture
from task_domain.test_service_integration import service_for

config = config_fixture


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    structured_response: dict


def test_sqlite_graph_rebuild_retains_original_messages_workspace_and_new_input(config):
    observed = []

    def build(checkpointer):
        graph = StateGraph(State)

        def step(state):
            humans = [m for m in state["messages"] if isinstance(m, HumanMessage)]
            observed.append([(m.id, m.content) for m in humans])
            return {"messages": [AIMessage(content="investigated")],
                    "structured_response": {"status": "needs_input", "summary": "investigated",
                                            "question": "Which input fails?"}}

        graph.add_node("investigate", step)
        graph.add_edge(START, "investigate")
        graph.add_edge("investigate", END)
        return graph.compile(checkpointer=checkpointer)

    with SqliteSaver.from_conn_string(str(config.database_path)) as saver:
        first = service_for(config, build(saver), workspace=True)
        task = first.start("repair parsing bug")
        definition = first.repository.get_definition(task.task_id)
    # Rebuild graph, saver and application service; only durable state is shared.
    with SqliteSaver.from_conn_string(str(config.database_path)) as saver:
        second = service_for(config, build(saver), workspace=True)
        resumed = second.continue_task(task.task_id, "empty string input")
        assert resumed.lifecycle is TaskLifecycleStatus.WAITING_INPUT
        assert second.repository.get_definition(task.task_id) == definition
        assert len(second.repository.list_runs(task.task_id)) == 2
    assert [text for _, text in observed[-1]] == ["repair parsing bug", "empty string input"]
    assert observed[0][0][0] == observed[1][0][0] == definition.original_message_id
