variable "aws_region" {
  description = "AWS region (e.g. us-east-1, us-west-2)"
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix used for resource names"
  type        = string
  default     = "clyde"
}

variable "instance_type" {
  description = "EC2 instance type. t3.micro is free-tier; t3.small is recommended for the simulator."
  type        = string
  default     = "t3.small"
}

variable "disk_gb" {
  description = "Root EBS volume size (gp3). 30 GB is free-tier max for 12 months."
  type        = number
  default     = 20
}

variable "ssh_pub_key" {
  description = "Your SSH public key (contents, not path). e.g. cat ~/.ssh/id_ed25519.pub"
  type        = string
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH. Set to your /32 for safety; default opens to the world."
  type        = string
  default     = "0.0.0.0/0"
}

variable "repo_url" {
  description = "HTTPS clone URL for the Clyde repo. Use a public URL or pre-bake credentials in cloud-init if private."
  type        = string
  default     = "https://github.com/raj-chinagundi/Clyde.git"
}

variable "repo_ref" {
  description = "Git ref (branch / tag / SHA) to deploy."
  type        = string
  default     = "main"
}

variable "domain" {
  description = "Optional public hostname. If set, Caddy auto-issues a Let's Encrypt cert. Point an A record at the EIP first."
  type        = string
  default     = ""
}

# ─── Secrets — gitignored via *.tfvars ───
variable "cerebras_api_key" {
  description = "CEREBRAS_API_KEY — empty string disables the provider."
  type        = string
  default     = ""
  sensitive   = true
}

variable "openrouter_api_key" {
  description = "OPENROUTER_API_KEY — alternative LLM provider."
  type        = string
  default     = ""
  sensitive   = true
}

variable "elevenlabs_api_key" {
  description = "CLYDE_ELEVENLABS_KEY — used by /api/tts narration."
  type        = string
  default     = ""
  sensitive   = true
}

variable "clyde_provider" {
  description = "Override CLYDE_LLM_PROVIDER (auto | cerebras | openrouter | mock)."
  type        = string
  default     = ""
}

variable "clyde_model" {
  description = "Override CLYDE_MODEL."
  type        = string
  default     = ""
}
