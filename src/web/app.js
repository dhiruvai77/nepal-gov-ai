"use strict";

const form = document.getElementById("ask-form");
const queryInput = document.getElementById("query");
const answerLanguage = document.getElementById("answer-language");
const documentFilter = document.getElementById("document-filter");
const evidenceLanguage = document.getElementById("evidence-language");
const submitButton = document.getElementById("submit-button");
const requestProgress = document.getElementById("request-progress");

const serviceStatus = document.getElementById("service-status");

const answerPanel = document.getElementById("answer-panel");
const answerStatus = document.getElementById("answer-status");
const acceptanceChip = document.getElementById("acceptance-chip");
const answerText = document.getElementById("answer-text");

const modelDiagnostic = document.getElementById("model-diagnostic");
const contextDiagnostic = document.getElementById("context-diagnostic");
const timeDiagnostic = document.getElementById("time-diagnostic");
const requestDiagnostic = document.getElementById("request-diagnostic");

const sourcesSection = document.getElementById("sources-section");
const sourceCount = document.getElementById("source-count");
const sourceList = document.getElementById("source-list");

const errorPanel = document.getElementById("error-panel");
const errorMessage = document.getElementById("error-message");

const exampleButtons = document.querySelectorAll(".example-button");

let progressTimer = null;


function setServiceStatus(status, text) {
    serviceStatus.classList.remove(
        "status-checking",
        "status-ready",
        "status-unavailable"
    );

    serviceStatus.classList.add(status);
    serviceStatus.textContent = text;
}


async function checkReadiness() {
    setServiceStatus(
        "status-checking",
        "Checking service…"
    );

    try {
        const response = await fetch(
            "/health/ready",
            {
                method: "GET",
                headers: {
                    "Accept": "application/json"
                }
            }
        );

        if (!response.ok) {
            setServiceStatus(
                "status-unavailable",
                "Service not ready"
            );
            return;
        }

        const body = await response.json();

        if (body.status === "ready") {
            setServiceStatus(
                "status-ready",
                "Service ready"
            );
            return;
        }

        setServiceStatus(
            "status-unavailable",
            "Service not ready"
        );

    } catch (error) {
        setServiceStatus(
            "status-unavailable",
            "Service unavailable"
        );
    }
}


function synchronizeDocumentLanguage() {
    const selectedOption =
        documentFilter.options[
            documentFilter.selectedIndex
        ];

    if (
        selectedOption
        && selectedOption.value
        && selectedOption.dataset.language
    ) {
        evidenceLanguage.value =
            selectedOption.dataset.language;

        evidenceLanguage.disabled = true;
        return;
    }

    evidenceLanguage.disabled = false;
}


function buildFilters() {
    const filters = {};

    const selectedOption =
        documentFilter.options[
            documentFilter.selectedIndex
        ];

    if (
        selectedOption
        && selectedOption.value
    ) {
        filters.document_id =
            selectedOption.value;

        if (selectedOption.dataset.language) {
            filters.language =
                selectedOption.dataset.language;
        }

        return filters;
    }

    if (evidenceLanguage.value) {
        filters.language =
            evidenceLanguage.value;
    }

    return filters;
}


function clearPreviousResult() {
    answerPanel.hidden = true;
    answerPanel.classList.remove("withheld");

    errorPanel.hidden = true;
    errorMessage.textContent = "";

    answerText.textContent = "";
    sourceList.replaceChildren();
    sourcesSection.hidden = true;

    modelDiagnostic.textContent = "";
    contextDiagnostic.textContent = "";
    timeDiagnostic.textContent = "";
    requestDiagnostic.textContent = "";
}


function startProgressTimer() {
    const startedAt = performance.now();

    requestProgress.textContent =
        "Retrieving evidence and generating an answer… 0s";

    progressTimer = window.setInterval(
        () => {
            const elapsedSeconds = Math.floor(
                (performance.now() - startedAt) / 1000
            );

            requestProgress.textContent =
                "Retrieving evidence and generating an answer… "
                + `${elapsedSeconds}s`;
        },
        1000
    );

    return startedAt;
}


function stopProgressTimer() {
    if (progressTimer !== null) {
        window.clearInterval(progressTimer);
        progressTimer = null;
    }

    requestProgress.textContent = "";
}


function formatPages(source) {
    if (source.page_start === source.page_end) {
        return `p. ${source.page_start}`;
    }

    return `pp. ${source.page_start}–${source.page_end}`;
}


function renderSources(sources) {
    sourceList.replaceChildren();

    if (!Array.isArray(sources) || sources.length === 0) {
        sourcesSection.hidden = true;
        return;
    }

    for (const source of sources) {
        const card = document.createElement("article");
        card.className = "source-card";

        const topLine = document.createElement("div");
        topLine.className = "source-topline";

        const evidenceLabel = document.createElement("span");
        evidenceLabel.className = "evidence-label";
        evidenceLabel.textContent = `[${source.evidence_id}]`;

        const pageLabel = document.createElement("span");
        pageLabel.className = "page-label";
        pageLabel.textContent = formatPages(source);

        topLine.append(
            evidenceLabel,
            pageLabel
        );

        const title = document.createElement("h4");
        title.textContent = source.title;

        const organization = document.createElement("p");
        organization.textContent =
            `${source.organization} · ${source.language.toUpperCase()}`;

        const documentId = document.createElement("p");
        documentId.textContent =
            `Document: ${source.document_id}`;

        const link = document.createElement("a");
        link.className = "source-link";
        link.href = source.source_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "Open official source";

        card.append(
            topLine,
            title,
            organization,
            documentId,
            link
        );

        sourceList.append(card);
    }

    sourceCount.textContent =
        `${sources.length} cited source`
        + (
            sources.length === 1
                ? ""
                : "s"
        );

    sourcesSection.hidden = false;
}


function renderAnswer(body, durationMs) {
    answerPanel.hidden = false;

    answerText.textContent =
        body.answer_text || "";

    if (body.accepted) {
        answerPanel.classList.remove("withheld");
        answerStatus.textContent = "Grounded response";

        acceptanceChip.className =
            "status-chip status-ready";

        acceptanceChip.textContent =
            "Accepted";
    } else {
        answerPanel.classList.add("withheld");
        answerStatus.textContent =
            "Insufficient supported evidence";

        acceptanceChip.className =
            "status-chip status-unavailable";

        acceptanceChip.textContent =
            "Withheld";
    }

    if (body.provider && body.model) {
        modelDiagnostic.textContent =
            `Model: ${body.provider}/${body.model}`;
    } else {
        modelDiagnostic.textContent =
            "Model: generation skipped";
    }

    contextDiagnostic.textContent =
        `Selected context: ${body.selected_context_count}`;

    timeDiagnostic.textContent =
        `Browser round trip: ${(durationMs / 1000).toFixed(1)}s`;

    requestDiagnostic.textContent =
        `Request: ${body.request_id}`;

    renderSources(body.sources);
}


function normalizeErrorDetail(detail) {
    if (typeof detail === "string") {
        return detail;
    }

    if (Array.isArray(detail)) {
        const messages = detail
            .map(
                (item) => {
                    if (
                        item
                        && typeof item.msg === "string"
                    ) {
                        return item.msg;
                    }

                    return null;
                }
            )
            .filter(Boolean);

        if (messages.length > 0) {
            return messages.join(" ");
        }
    }

    return "The request could not be completed.";
}


function showError(message) {
    errorMessage.textContent = message;
    errorPanel.hidden = false;
}


async function submitQuestion(event) {
    event.preventDefault();

    const query = queryInput.value.trim();

    if (!query) {
        showError("Enter a question before submitting.");
        queryInput.focus();
        return;
    }

    clearPreviousResult();

    submitButton.disabled = true;
    submitButton.textContent = "Working…";

    const startedAt = startProgressTimer();

    const payload = {
        query: query,
        answer_language: answerLanguage.value
    };

    const filters = buildFilters();

    if (Object.keys(filters).length > 0) {
        payload.filters = filters;
    }

    try {
        const response = await fetch(
            "/v1/answer",
            {
                method: "POST",
                headers: {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            }
        );

        let body = null;

        try {
            body = await response.json();
        } catch (error) {
            body = null;
        }

        if (!response.ok) {
            const detail =
                body && Object.hasOwn(body, "detail")
                    ? body.detail
                    : null;

            throw new Error(
                normalizeErrorDetail(detail)
            );
        }

        const durationMs =
            performance.now() - startedAt;

        renderAnswer(
            body,
            durationMs
        );

        answerPanel.scrollIntoView(
            {
                behavior: "smooth",
                block: "start"
            }
        );

    } catch (error) {
        const message =
            error instanceof Error
                ? error.message
                : "The request could not be completed.";

        showError(message);

    } finally {
        stopProgressTimer();

        submitButton.disabled = false;
        submitButton.textContent = "Ask NepalGov AI";
    }
}


for (const button of exampleButtons) {
    button.addEventListener(
        "click",
        () => {
            queryInput.value =
                button.dataset.query || "";

            answerLanguage.value =
                button.dataset.language || "en";

            documentFilter.value =
                button.dataset.document || "";

            synchronizeDocumentLanguage();

            queryInput.focus();
        }
    );
}


documentFilter.addEventListener(
    "change",
    synchronizeDocumentLanguage
);

form.addEventListener(
    "submit",
    submitQuestion
);


synchronizeDocumentLanguage();
checkReadiness();