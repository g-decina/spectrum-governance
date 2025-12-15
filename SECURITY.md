# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

**git@lexiconautomata.com**

Include the following information:

1. **Description**: A clear description of the vulnerability
2. **Impact**: What an attacker could achieve by exploiting this
3. **Steps to reproduce**: Detailed steps to reproduce the issue
4. **Affected versions**: Which versions are affected
5. **Suggested fix**: If you have one (optional)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Initial assessment**: We will provide an initial assessment within 7 days
- **Resolution timeline**: We aim to resolve critical issues within 30 days
- **Credit**: We will credit you in the security advisory (unless you prefer anonymity)

### Scope

This security policy covers:

- The `spectrum` Python package
- The `spectrum-core` Rust crates
- Official documentation and examples

Out of scope:

- Third-party dependencies (report to their maintainers)
- Issues in example/demo code that don't affect the library

## Security Considerations for Users

### Model Security

Spectrum Governance is designed to audit ML models. When using this library:

1. **Adversarial testing**: The Red Team module generates adversarial examples. Ensure you have authorization before testing models you don't own.

2. **Sensitive data**: Audit reports may contain sensitive information about model behavior. Handle reports according to your organization's data policies.

3. **Model artifacts**: When loading models (ONNX, pickle, etc.), only load models from trusted sources. Malicious model files can execute arbitrary code.

### Dependency Security

We regularly update dependencies to address known vulnerabilities. Run:

```bash
pip install --upgrade spectrum-governance
```

To check for known vulnerabilities in your environment:

```bash
pip install safety
safety check
```

## Security Best Practices

When deploying Spectrum Governance in production:

1. **Isolate execution**: Run adversarial tests in isolated environments
2. **Audit logging**: Enable OpenLineage logging for audit trails
3. **Access control**: Restrict access to audit reports containing model details
4. **Update regularly**: Keep the library and dependencies updated
