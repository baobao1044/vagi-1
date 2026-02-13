use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use regex::Regex;

use crate::KernelContext;
use crate::models::{
    ErrorResponse, HealthResponse, InitStateRequest, JitExecuteRequest, JitExecuteResponse,
    HdcTemplateBindRequest, HdcTemplateBindResponse, HdcTemplateQueryRequest,
    HdcTemplateQueryResponse, HdcTemplateUpsertRequest, HdcTemplateUpsertResponse,
    HdcEvolutionMutateRequest, HdcEvolutionMutateResponse,
    HdcWeaveExecuteRequest, HdcWeaveExecuteResponse, HdcWeavePlanCandidate,
    HdcWeavePlanRequest, HdcWeavePlanResponse, SnapshotRequest, SnapshotResponse,
    UpdateStateRequest, VerifierRequest, VerifierResponse, WorldSimulateRequest,
    WorldSimulateResponse,
};

const DEFAULT_WEAVE_EXECUTE_RISK_THRESHOLD: f32 = 0.65;
const INTERNAL_TOKEN_HEADER: &str = "x-vagi-internal-token";
const MAX_BINDING_VALUE_BYTES: usize = 64;
const MAX_WEAVE_QUERY_BYTES: usize = 2_048;
const MAX_ACTION_BYTES: usize = 8_192;

pub fn build_router(ctx: Arc<KernelContext>, internal_token: String) -> Router {
    let internal_routes = Router::new()
        .route("/state/init", post(init_state))
        .route("/state/update", post(update_state))
        .route("/state/{session_id}", get(get_state))
        .route("/state/snapshot", post(snapshot_state))
        .route("/world/simulate", post(simulate_world))
        .route("/verifier/check", post(verify_patch))
        .route("/jit/execute", post(execute_jit))
        .route("/hdc/templates/upsert", post(hdc_upsert_template))
        .route("/hdc/templates/query", post(hdc_query_templates))
        .route("/hdc/templates/bind", post(hdc_bind_template))
        .route("/hdc/evolution/mutate", post(hdc_evolution_mutate))
        .route("/hdc/weave/execute", post(hdc_weave_execute))
        .route("/hdc/weave/plan", post(hdc_weave_plan))
        .layer(middleware::from_fn_with_state(
            internal_token,
            require_internal_token,
        ));

    Router::new()
        .route("/healthz", get(healthz))
        .nest("/internal", internal_routes)
        .with_state(ctx)
}

async fn healthz(State(ctx): State<Arc<KernelContext>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        hidden_size: ctx.state_manager.hidden_size(),
    })
}

async fn init_state(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<InitStateRequest>,
) -> Json<crate::models::HiddenState> {
    Json(ctx.state_manager.init_session(request.session_id))
}

async fn update_state(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<UpdateStateRequest>,
) -> Result<Json<crate::models::HiddenState>, ApiError> {
    let state = ctx
        .state_manager
        .update_state(&request.session_id, &request.input)
        .map_err(ApiError::bad_request)?;
    Ok(Json(state))
}

async fn get_state(
    State(ctx): State<Arc<KernelContext>>,
    Path(session_id): Path<String>,
) -> Result<Json<crate::models::HiddenState>, ApiError> {
    let Some(state) = ctx.state_manager.get_state(&session_id) else {
        return Err(ApiError::not_found(format!(
            "session_id `{session_id}` does not exist"
        )));
    };
    Ok(Json(state))
}

async fn snapshot_state(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<SnapshotRequest>,
) -> Result<Json<SnapshotResponse>, ApiError> {
    let Some(state) = ctx.state_manager.get_state(&request.session_id) else {
        return Err(ApiError::not_found(format!(
            "session_id `{}` does not exist",
            request.session_id
        )));
    };
    let epoch = request.epoch.unwrap_or(state.step);
    let key = ctx
        .snapshot_store
        .save_state(&state, epoch)
        .map_err(ApiError::internal)?;
    Ok(Json(SnapshotResponse {
        key,
        step: state.step,
        checksum: state.checksum,
    }))
}

async fn simulate_world(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<WorldSimulateRequest>,
) -> Result<Json<WorldSimulateResponse>, ApiError> {
    validate_text_size("action", &request.action, MAX_ACTION_BYTES)?;
    let _sid = request.session_id;
    Ok(Json(ctx.world_model.simulate(&request.action)))
}

async fn verify_patch(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<VerifierRequest>,
) -> Json<VerifierResponse> {
    Json(ctx.verifier.check(&request))
}

async fn execute_jit(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<JitExecuteRequest>,
) -> Result<Json<JitExecuteResponse>, ApiError> {
    let response = ctx
        .jit_engine
        .compile_and_execute(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_upsert_template(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcTemplateUpsertRequest>,
) -> Result<Json<HdcTemplateUpsertResponse>, ApiError> {
    let response = ctx
        .hdc_memory
        .upsert_template(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_query_templates(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcTemplateQueryRequest>,
) -> Result<Json<HdcTemplateQueryResponse>, ApiError> {
    let response = ctx
        .hdc_memory
        .query_templates(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_bind_template(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcTemplateBindRequest>,
) -> Result<Json<HdcTemplateBindResponse>, ApiError> {
    validate_bindings(&request.bindings)?;
    let response = ctx
        .hdc_memory
        .bind_template(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_evolution_mutate(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcEvolutionMutateRequest>,
) -> Result<Json<HdcEvolutionMutateResponse>, ApiError> {
    let response = ctx
        .mutation_engine
        .evolve_templates(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_weave_execute(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcWeaveExecuteRequest>,
) -> Result<Json<HdcWeaveExecuteResponse>, ApiError> {
    validate_text_size("query", &request.query, MAX_WEAVE_QUERY_BYTES)?;
    validate_bindings(&request.bindings)?;

    let query_response = ctx
        .hdc_memory
        .query_templates(&HdcTemplateQueryRequest {
            query: request.query.clone(),
            top_k: Some(request.top_k.unwrap_or(1).clamp(1, 5)),
        })
        .map_err(ApiError::bad_request)?;

    let best = query_response
        .hits
        .first()
        .ok_or_else(|| ApiError::bad_request("no template found for query"))?;

    let bind_response = ctx
        .hdc_memory
        .bind_template(&HdcTemplateBindRequest {
            template_id: best.template_id.clone(),
            bindings: request.bindings.clone(),
        })
        .map_err(ApiError::bad_request)?;

    let verifier_response = ctx.verifier.check(&VerifierRequest {
        patch_ir: bind_response.bound_logic.clone(),
        max_loop_iters: None,
        side_effect_budget: Some(3),
        timeout_ms: Some(80),
    });
    if !verifier_response.pass {
        return Err(ApiError::unprocessable_with_violations(
            "weave_execute rejected by verifier".to_string(),
            verifier_response.violations,
        ));
    }

    let simulation = ctx.world_model.simulate(&bind_response.bound_logic);
    if simulation.risk_score > DEFAULT_WEAVE_EXECUTE_RISK_THRESHOLD {
        return Err(ApiError::unprocessable_with_risk(
            format!(
                "weave_execute rejected by risk gate: {:.2} > {:.2}",
                simulation.risk_score, DEFAULT_WEAVE_EXECUTE_RISK_THRESHOLD
            ),
            simulation.risk_score,
            DEFAULT_WEAVE_EXECUTE_RISK_THRESHOLD,
            vec![format!(
                "risk_score_exceeded:{:.2}>{:.2}",
                simulation.risk_score, DEFAULT_WEAVE_EXECUTE_RISK_THRESHOLD
            )],
        ));
    }

    let jit_response = ctx
        .jit_engine
        .compile_and_execute(&JitExecuteRequest {
            logic: bind_response.bound_logic.clone(),
            input: request.input,
        })
        .map_err(ApiError::bad_request)?;

    Ok(Json(HdcWeaveExecuteResponse {
        template_id: best.template_id.clone(),
        similarity: best.similarity,
        bound_logic: bind_response.bound_logic,
        output: jit_response.output,
        backend: jit_response.backend,
        compile_micros: jit_response.compile_micros,
        execute_micros: jit_response.execute_micros,
    }))
}

async fn hdc_weave_plan(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcWeavePlanRequest>,
) -> Result<Json<HdcWeavePlanResponse>, ApiError> {
    validate_text_size("query", &request.query, MAX_WEAVE_QUERY_BYTES)?;
    validate_bindings(&request.bindings)?;

    let query_response = ctx
        .hdc_memory
        .query_templates(&HdcTemplateQueryRequest {
            query: request.query.clone(),
            top_k: Some(request.top_k.unwrap_or(3).clamp(1, 10)),
        })
        .map_err(ApiError::bad_request)?;

    let verifier_required = request.verifier_required.unwrap_or(true);
    let risk_threshold = request.risk_threshold.unwrap_or(0.65).clamp(0.01, 0.99);

    let mut candidates = Vec::new();
    for hit in &query_response.hits {
        let mut candidate = HdcWeavePlanCandidate {
            template_id: hit.template_id.clone(),
            similarity: hit.similarity,
            bound_logic: String::new(),
            output: 0,
            verifier_pass: false,
            verifier_violations: Vec::new(),
            risk_score: 1.0,
            confidence: 0.0,
            compile_micros: 0,
            execute_micros: 0,
            accepted: false,
            rejection_reason: None,
        };

        match ctx.hdc_memory.bind_template(&HdcTemplateBindRequest {
            template_id: hit.template_id.clone(),
            bindings: request.bindings.clone(),
        }) {
            Ok(bind_response) => {
                candidate.bound_logic = bind_response.bound_logic.clone();

                match ctx.jit_engine.compile_and_execute(&JitExecuteRequest {
                    logic: bind_response.bound_logic.clone(),
                    input: request.input,
                }) {
                    Ok(jit_response) => {
                        candidate.output = jit_response.output;
                        candidate.compile_micros = jit_response.compile_micros;
                        candidate.execute_micros = jit_response.execute_micros;

                        let sim = ctx.world_model.simulate(&bind_response.bound_logic);
                        candidate.risk_score = sim.risk_score;
                        candidate.confidence = sim.confidence;

                        let verify = ctx.verifier.check(&VerifierRequest {
                            patch_ir: bind_response.bound_logic,
                            max_loop_iters: Some(2_048),
                            side_effect_budget: Some(3),
                            timeout_ms: Some(80),
                        });
                        candidate.verifier_pass = verify.pass;
                        candidate.verifier_violations = verify.violations;

                        let pass_gate =
                            (!verifier_required || candidate.verifier_pass)
                                && candidate.risk_score <= risk_threshold;
                        candidate.accepted = pass_gate;
                        if !pass_gate {
                            candidate.rejection_reason = Some(format!(
                                "verifier_pass={} risk_score={:.2} threshold={:.2}",
                                candidate.verifier_pass, candidate.risk_score, risk_threshold
                            ));
                        }
                    }
                    Err(err) => {
                        candidate.rejection_reason =
                            Some(format!("jit_compile_or_execute_failed:{err}"));
                    }
                }
            }
            Err(err) => {
                candidate.rejection_reason = Some(format!("template_bind_failed:{err}"));
            }
        }

        candidates.push(candidate);
    }

    candidates.sort_by(|a, b| {
        b.accepted
            .cmp(&a.accepted)
            .then_with(|| a.risk_score.total_cmp(&b.risk_score))
            .then_with(|| b.similarity.total_cmp(&a.similarity))
            .then_with(|| a.execute_micros.cmp(&b.execute_micros))
    });

    let selected_index = candidates.iter().position(|candidate| candidate.accepted);
    let selected_template_id = selected_index.map(|idx| candidates[idx].template_id.clone());

    Ok(Json(HdcWeavePlanResponse {
        selected_template_id,
        selected_index,
        candidates,
        backend: "wasmtime-cranelift-jit",
    }))
}

async fn require_internal_token(
    State(expected_token): State<String>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let provided = request
        .headers()
        .get(INTERNAL_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok());
    if !provided.is_some_and(|value| constant_time_eq(value, expected_token.as_str())) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(next.run(request).await)
}

fn constant_time_eq(lhs: &str, rhs: &str) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }
    let mut diff = 0u8;
    for (a, b) in lhs.as_bytes().iter().zip(rhs.as_bytes().iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    violations: Option<Vec<String>>,
    risk_score: Option<f32>,
    risk_threshold: Option<f32>,
}

impl ApiError {
    fn bad_request<E: ToString>(err: E) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: err.to_string(),
            violations: None,
            risk_score: None,
            risk_threshold: None,
        }
    }

    fn not_found(message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message,
            violations: None,
            risk_score: None,
            risk_threshold: None,
        }
    }

    fn internal<E: ToString>(err: E) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
            violations: None,
            risk_score: None,
            risk_threshold: None,
        }
    }

    fn unprocessable_with_violations(message: String, violations: Vec<String>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message,
            violations: Some(violations),
            risk_score: None,
            risk_threshold: None,
        }
    }

    fn unprocessable_with_risk(
        message: String,
        risk_score: f32,
        risk_threshold: f32,
        violations: Vec<String>,
    ) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message,
            violations: Some(violations),
            risk_score: Some(risk_score),
            risk_threshold: Some(risk_threshold),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let payload = Json(ErrorResponse {
            error: self.message,
            violations: self.violations,
            risk_score: self.risk_score,
            risk_threshold: self.risk_threshold,
        });
        (self.status, payload).into_response()
    }
}

fn validate_bindings(bindings: &HashMap<String, String>) -> Result<(), ApiError> {
    let key_regex = Regex::new(r"^[A-Za-z0-9_-]{1,64}$").map_err(ApiError::internal)?;
    let numeric_regex = Regex::new(r"^-?\d+$").map_err(ApiError::internal)?;
    let safe_text_regex = Regex::new(r"^[A-Za-z0-9 _.,:/-]{1,64}$").map_err(ApiError::internal)?;

    for (key, raw_value) in bindings {
        if !key_regex.is_match(key) {
            return Err(ApiError::unprocessable_with_violations(
                format!("binding key `{key}` is invalid"),
                vec![format!("binding_key_invalid:{key}")],
            ));
        }

        let value = raw_value.trim();
        if value.is_empty() {
            return Err(ApiError::unprocessable_with_violations(
                format!("binding value for `{key}` must not be empty"),
                vec![format!("binding_value_empty:{key}")],
            ));
        }
        if value.len() > MAX_BINDING_VALUE_BYTES {
            return Err(ApiError::unprocessable_with_violations(
                format!("binding value for `{key}` exceeds size limit"),
                vec![format!("binding_value_too_large:{key}")],
            ));
        }

        if numeric_regex.is_match(value) {
            continue;
        }

        if !safe_text_regex.is_match(value) {
            return Err(ApiError::unprocessable_with_violations(
                format!("binding value for `{key}` contains unsafe characters"),
                vec![format!("binding_value_invalid:{key}")],
            ));
        }
    }

    Ok(())
}

fn validate_text_size(field: &str, value: &str, max_bytes: usize) -> Result<(), ApiError> {
    if value.is_empty() {
        return Err(ApiError::bad_request(format!("{field} must not be empty")));
    }
    if value.len() > max_bytes {
        return Err(ApiError::bad_request(format!(
            "{field} exceeds max size: {}>{max_bytes} bytes",
            value.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::Json;

    use crate::KernelContext;
    use crate::models::{HdcTemplateUpsertRequest, HdcWeaveExecuteRequest};

    use super::hdc_weave_execute;

    #[tokio::test]
    async fn weave_execute_rejects_logic_with_infinite_loop_pattern() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let snapshot_path = temp_dir.path().join("snapshots.redb");
        let ctx = Arc::new(KernelContext::new(&snapshot_path).expect("create kernel context"));

        ctx.hdc_memory
            .upsert_template(&HdcTemplateUpsertRequest {
                template_id: "danger_loop_template".to_string(),
                logic_template: "add 1\n# zx_danger_loop while(true)".to_string(),
                tags: vec!["zx_danger_loop".to_string()],
            })
            .expect("upsert dangerous template");

        let result = hdc_weave_execute(
            State(ctx),
            Json(HdcWeaveExecuteRequest {
                query: "zx_danger_loop".to_string(),
                input: 9,
                top_k: Some(1),
                bindings: HashMap::new(),
            }),
        )
        .await;

        let err = match result {
            Ok(_) => panic!("weave_execute should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
        let violations = err.violations.expect("violations should be present");
        assert!(violations.iter().any(|v| v == "infinite_loop_risk"));
    }
}
