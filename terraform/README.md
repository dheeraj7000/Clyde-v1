# Clyde — Terraform deployment

Single-instance EC2 setup. Provisions one Ubuntu 24.04 box with a static
Elastic IP, runs `python -m clyde.web` under `systemd`, and fronts it
with Caddy on `:80` (or `:443` with auto-LE if you supply a domain).

## Prereqs

- Terraform `>= 1.5`
- AWS CLI configured (`aws configure` or `AWS_PROFILE=…`)
- An SSH keypair locally: `ssh-keygen -t ed25519 -f ~/.ssh/clyde`

## Deploy

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — paste ssh_pub_key, fill in API keys
terraform init
terraform plan
terraform apply
```

Apply takes ~2 min for AWS, then ~3–5 min for cloud-init to finish
installing dependencies inside the box. Tail it from your laptop:

```bash
ssh ubuntu@$(terraform output -raw public_ip) 'sudo tail -f /var/log/cloud-init-output.log'
```

When you see `clyde.service: ActiveState=active`, browse to:

```bash
echo "Open: $(terraform output -raw url)"
```

## Sizing

| Instance | vCPU | RAM | Free tier? | Notes |
|---|---|---|---|---|
| `t3.micro` | 2 | 1 GiB | ✅ 750 h/mo · 12 mo | Cap `run_count ≤ 6` to avoid OOM during simulations. |
| `t3.small` | 2 | 2 GiB | ❌ | Default. Comfortable for `run_count = 24`. |
| `t3.medium` | 2 | 4 GiB | ❌ | Headroom for parallel branches. |

## Updating the app

The cloud-init step clones the repo at boot. To pick up new commits:

```bash
ssh ubuntu@$(terraform output -raw public_ip) <<'EOF'
  cd /opt/clyde/app
  git pull
  /opt/clyde/app/.venv/bin/pip install -e '.[web]'
  sudo systemctl restart clyde
EOF
```

To redeploy from a different `repo_ref` (a new branch, a tagged
release), bump the variable and run `terraform apply` — the instance
is replaced (`user_data_replace_on_change = true`), so the EIP detaches
briefly and reattaches after the new instance is up.

## TLS / custom domain

1. Set `domain = "clyde.example.com"` in `terraform.tfvars`.
2. `terraform apply` to roll the Caddyfile.
3. Point an A record at the EIP (`terraform output public_ip`).
4. Caddy auto-issues a Let's Encrypt cert on first request to that hostname.

If DNS hasn't propagated, Caddy retries on a backoff. Watch `journalctl -u caddy -f`.

## Tear down

```bash
terraform destroy
```

This releases the Elastic IP. **Important**: an unattached EIP is
billed ~$0.005/h. If you only stop the instance without destroying
the EIP, you'll keep accruing charges.

## Cost (us-east-1, indicative)

| Resource | Hourly | Monthly | Notes |
|---|---:|---:|---|
| `t3.small` | $0.0208 | $15.18 | always-on |
| `t3.micro` | $0.0104 | $7.59 | covered 12 mo by free tier |
| 20 GB gp3 EBS | — | $1.60 | $0.08/GB-mo |
| Elastic IP (attached) | — | $0.00 | free while attached |
| Egress | — | $0.00 | first 100 GB/mo always free |

## Security notes

- `ssh_cidr` defaults to `0.0.0.0/0` — **tighten to your `/32`** for production.
- IMDSv2 is enforced (`http_tokens = "required"`).
- `.env` lives at `/opt/clyde/.env`, mode `0600`, owned by `ubuntu`.
- Caddy upgrades to TLS 1.2+ by default; auto-LE handles cert rotation.
- Rotate the ElevenLabs / Cerebras / OpenRouter keys after the demo —
  they pass through `terraform.tfvars` (gitignored) but exist in the
  Terraform state file. Treat the state as sensitive.
